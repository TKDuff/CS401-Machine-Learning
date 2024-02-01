### A Pluto.jl notebook ###
# v0.19.35

using Markdown
using InteractiveUtils

# ╔═╡ 05840afa-cc0e-442c-8869-0dac89e1f2c9
begin
	import Pkg
    Pkg.activate(mktempdir())
	Pkg.develop(url="https://github.com/rgreilly/Transformers")    
	Pkg.add(["Revise", "PlutoUI", "Flux", "DataFrames", "Printf", 
		"BSON", "JSON", "Arrow", "StatsBase", "Unicode", "Random", 
		"DataStructures", "ProgressMeter", "RemoteFiles"])
	
	using Revise
	using TransformersLite
	using PlutoUI
	using Flux
	using Flux.CUDA
	using Flux: DataLoader
	using DataFrames
	using BSON, JSON
	using Arrow
	using Printf
	using StatsBase
	using StatsBase: mean
	using Dates
	using Unicode
	using Random
	using DataStructures
	using TransformersLite
	using RemoteFiles
end;

# ╔═╡ f56c57c7-6301-4c07-a236-c11276eb1f32
md"""
![transformer](https://github.com/rgreilly/images/assets/1801654/f5d27ba2-5f19-4b53-8cdf-17defc3a910b)
"""

# ╔═╡ c43f57ce-90c5-4264-9e60-89f676bda905
TableOfContents()

# ╔═╡ fefcdcfc-589e-482c-acc9-1e1a95f8d622
md"""
# Sentiment analysis using a transformer
## Model training
"""

# ╔═╡ f2ac80eb-f36d-4925-8ce4-c6a8dc8d42b6
md"""
!!! tip
	Hidden below is a useful snippet of HTML to setup a `restart` button in case training gets out of hand.
"""

# ╔═╡ 39ff3330-99b0-4cde-b9ba-9ec2d472dfda
HTML("""
<!-- the wrapper span -->
<div>
	<button id="myrestart" href="#">Restart</button>
	
	<script>
		const div = currentScript.parentElement
		const button = div.querySelector("button#myrestart")
		const cell= div.closest('pluto-cell')
		console.log(button);
		button.onclick = function() { restart_nb() };
		function restart_nb() {
			console.log("Restarting Notebook");
		        cell._internal_pluto_actions.send(                    
		            "restart_process",
                            {},
                            {
                                notebook_id: editor_state.notebook.notebook_id,
                            }
                        )
		};
	</script>
</div>
""")

# ╔═╡ dc4d3b2d-f583-47f4-8368-52449829fb22
md"""
### Download module from GitHub
"""

# ╔═╡ 8154b53a-09b9-4738-af9a-ffebf89a9e88
md"""
Because we're working from a GitHub repo and not the standard Julia repository, we  have to manage the installation and use of all packages rather than rely on Pluto.
"""

# ╔═╡ 912c1b67-8ba7-40f1-a4cf-33eaa4b6d7c6
md"""
In addition to the list of modules, we also need to include individual Julia files from the repo.  This is done using the `RemoteFiles` module.  However, this downloads them as `JSON` objects, which we need to convert back to regular `.jl` files.
"""

# ╔═╡ 89124a6a-ecad-4570-86b0-f4e927e6fff0
begin
	@RemoteFileSet FILES "Transformer utilities" begin

		utilities = @RemoteFile "https://github.com/rgreilly/Transformers/blob/main/examples/utilities.jl" dir="utilities" file="utilities.jl.json"
			
		training = @RemoteFile "https://github.com/rgreilly/Transformers/blob/main/examples/training.jl" dir="utilities" file="training.jl.json"
	end

	download(FILES) # Files downloaded in JSON format
end

# ╔═╡ de110894-6faf-45e9-854d-1da663230472
function convertJSON(inFile, outFile)
	body = JSON.parsefile(inFile)["payload"]["blob"]["rawLines"]
	open(outFile, "w") do f
	  for i in body
	    println(f, i)
	  end
	end 
end

# ╔═╡ 908065d1-6c48-4352-a34c-8d4d972799ff
begin
	convertJSON("utilities/utilities.jl.json", "utilities/utilities.jl")
	convertJSON("utilities/training.jl.json", "utilities/training.jl")
	
	include("utilities/utilities.jl")
	include("utilities/training.jl")
end;

# ╔═╡ f8deace6-2bf9-49c5-98c0-1c18aca95aff
md"""
## Setup the training data
"""

# ╔═╡ 359798d2-269b-4ae8-ae2d-e9fd3a26cf7e
md"""
 - Setup the file path to the Kaggle Amazon reviews dataset
 - Assign values to various hyper-paraemeters and store them in a `dictionary`.
 - Set number of training epochs
"""

# ╔═╡ 1072107e-e99b-46a9-ad10-9f6f894ad0da
md"""
!!! tip
	Here is where you can manipulate various training parameters - `pdrop`: proportion of weights to dropout (i.e., set to zero); `dim-embedding`: size of embedding; `n_epoch`: number of epochs.
"""

# ╔═╡ cd40a8a8-78f6-43f1-84c2-5664dbab402e
begin
	path = normpath(joinpath(@__DIR__, "..", "examples/datasets", "amazon_reviews_multi", "en", "1.0.0"))
	filename = "train.arrow"
	to_device = cpu # gpu or cpu
	
	filepath = joinpath(path, filename)
	
	df = DataFrame(Arrow.Table(filepath))
	display(first(df, 20))
	println("")
	
	hyperparameters = Dict(
	    "seed" => 314159,
	    "tokenizer" => "none", # options: none bpe affixes
	    "nlabels" => 5,
	    "pdrop" => 0.1,
	    "dim_embedding" => 32
	)
	nlabels = hyperparameters["nlabels"]
	n_epochs = 10
end;

# ╔═╡ 5e56420c-a137-47b0-b2b7-e2c8c4341571
md"""
## Tokenisers
"""

# ╔═╡ 9e4ad8a2-5a85-492e-bd9d-0f0f3f5e87d3
md"""
Select a tokeniser.  In this case, `none`, which just uses the various inflected word forms.
"""

# ╔═╡ e700c3f1-4ff1-46db-b6db-ad274f6ecb8a
begin
	if hyperparameters["tokenizer"] == "bpe"
	    directory = joinpath("vocab", "bpe")
	    path_rules = joinpath(directory, "amazon_reviews_train_en_rules.txt")
	    path_vocab = joinpath(directory, "amazon_reviews_train_en_vocab.txt")
	    tokenizer = load_bpe(path_rules, startsym="⋅")
	elseif hyperparameters["tokenizer"] == "affixes"
	    directory = joinpath("vocab","affixes")
	    path_vocab = joinpath(directory, "amazon_reviews_train_en_vocab.txt")
	    tokenizer = load_affix_tokenizer(path_vocab)
	elseif hyperparameters["tokenizer"] == "none"
	    path_vocab = joinpath("vocab", "amazon_reviews_train_en.txt")
	    tokenizer = identity
	end
	
	vocab = load_vocab(joinpath(@__DIR__, path_vocab))
	indexer = IndexTokenizer(vocab, "[UNK]")
	
	display(tokenizer)
	println("")
	display(indexer)
	println("")
	
end

# ╔═╡ bc01f007-7500-4fed-afd6-47df14009859
md"""
## Tokenise
"""

# ╔═╡ a4497815-598f-4ebe-a412-c3d145113f07
md"""
Extract the review body and star rating from the dataframe and create embeddings.  Partition data into training and validation sets.
"""

# ╔═╡ b3d4bd35-213d-453b-9f7e-1d56f4202199
begin
	documents = df[!, :review_body]
	labels = df[!, :stars]
	max_length = 50
	indices_path = joinpath(@__DIR__, "outputs", "indices_" * hyperparameters["tokenizer"] * ".bson")
	@time tokens = map(d->preprocess(d, tokenizer, max_length=max_length), documents)
	@time indices = indexer(tokens)
	
	y_labels = Int.(labels)
	if nlabels == 1
	    y_labels[labels .≤ 2] .= 0
	    y_labels[labels .≥ 4] .= 1
	    idxs = labels .!= 3
	    y_labels = reshape(y_labels, 1, :)
	else
	    idxs = Base.OneTo(length(labels))
	    y_labels = Flux.onehotbatch(y_labels, 1:nlabels)
	end
	
	X_train, y_train = indices[:, idxs], y_labels[:, idxs];
	rng = MersenneTwister(hyperparameters["seed"])
	train_data, val_data = split_validation(X_train, y_train; rng=rng)
	
	println("train samples:      ", size(train_data[1]), " ", size(train_data[2]))
	println("validation samples: ", size(val_data[1]), " ", size(val_data[2]))
	println("")
end

# ╔═╡ d25d18cb-49e3-4c89-b729-07bcf5b33663
md"""
## Model definition
"""

# ╔═╡ cf012cc8-eda9-4487-8c38-8b8e80cc5984
md"""
Assemble the model's components.
"""

# ╔═╡ 27dfaed4-a9d6-4c34-b25d-188607d53c20
md"""
!!! tip
	Here's where you might want to adjust the number and nature of the `encoder` blocks (e.g., attention heads, dropout), number of `Dense` layers and their characteristics (e.g., activation function, dimensions), the number of `dropout` layers.
"""

# ╔═╡ 07d3c4ec-de35-40b0-b31c-ecaedaae622a
begin
	dim_embedding = hyperparameters["dim_embedding"]
	pdrop = hyperparameters["pdrop"]
	model = TransformersLite.TransformerClassifier(
	    Embed(dim_embedding, length(indexer)), 
	    PositionEncoding(dim_embedding), 
	    Dropout(pdrop),
	    TransformerEncoderBlock[
	        TransformerEncoderBlock(4, dim_embedding, dim_embedding * 4; pdrop=pdrop)
	    ],
	    Dense(dim_embedding, 1), 
	    FlattenLayer(),
	    Dense(max_length, nlabels)
	    )
	display(model)
	println("")
	model = to_device(model) 
	
	hyperparameters["model"] = "$(typeof(model).name.wrapper)"
	hyperparameters["trainable parameters"] = sum(length, Flux.params(model));
	
	if nlabels == 1
	    loss(x, y) = Flux.logitbinarycrossentropy(x, y)
	    accuracy(ŷ, y) = mean((Flux.sigmoid.(ŷ) .> 0.5) .== y)
	else
	    loss(x, y) = Flux.logitcrossentropy(x, y)
	    accuracy(ŷ, y) = mean(Flux.onecold(ŷ) .== Flux.onecold(y))
	end
end;

# ╔═╡ da9d022e-dcde-42a4-af0a-a649dd3a744b
md"""
## Training
"""

# ╔═╡ da3367b9-80e7-4eba-b4e9-092541332645
md"""
 - Setup the dataloaders to batch and shuffle the training and validation data.
 - Print out initial accuracy and loss values for the validation data. 
 - Setup a sub-directory in the outputs directory, based on date and time, to store the trained model and associated hyperparameters.
 - call the `train!` method and log training progress.
"""

# ╔═╡ 44f5b65a-41f7-4e2a-946c-942db95214a6
begin
	opt_state = Flux.setup(Adam(), model)
	batch_size = 32
	
	train_data_loader = DataLoader(train_data |> to_device; batchsize=batch_size,
		shuffle=true)
	val_data_loader = DataLoader(val_data |> to_device; batchsize=batch_size,
		shuffle=false)
	
	val_acc = batched_metric(model, accuracy, val_data_loader)
	val_loss = batched_metric(model, loss, val_data_loader)
	
	@printf "val_acc=%.4f%% ; " val_acc * 100
	@printf "val_loss=%.4f \n" val_loss
	println("")
	
	directory2 = normpath( joinpath(@__DIR__, "..", "outputs", 
		Dates.format(now(), "yyyymmdd_HHMM")))
	mkpath(directory2)
	output_path = joinpath(directory2, "model.bson")
	history_path = joinpath(directory2, "history.json")
	
	hyperparameter_path = joinpath(directory2, "hyperparameters.json")
	open(hyperparameter_path, "w") do f
	    JSON.print(f, hyperparameters)
	end
	println("saved hyperparameters to $(hyperparameter_path).")
	println("")
	
	start_time = time_ns()
	history = train!(
	    loss, model, train_data_loader, opt_state, val_data_loader;
			num_epochs=n_epochs)
	end_time = time_ns() - start_time
	
	println("done training")
	@printf "time taken: %.2fs\n" end_time/1e9
end

# ╔═╡ 1c7b43ea-9c2a-4596-9d2a-340f9731df63
accuracy

# ╔═╡ fa8bf210-437f-4bc9-81a1-31e30c10edb8
md"""
## Save the model
"""

# ╔═╡ cbffc631-9865-4b0c-a182-eb82c2420e90
md"""
Save model, embeddings, and training history to the `outputs` sub-directory.
"""

# ╔═╡ 96c1c5fe-9abc-11ee-2316-4b1e5233a493
begin
	model2 = model |> cpu
	if hasproperty(tokenizer, :cache)
	    # empty cache
	    tokenizer2 = similar(tokenizer)
	end
	BSON.bson(
	    output_path, 
	    Dict(
	        :model=> model2, 
	        :tokenizer=>tokenizer,
	        :indexer=>indexer
	    )
	)
	println("saved model to $(output_path).")
	
	open(history_path,"w") do f
	  JSON.print(f, history)
	end
	println("saved history to $(history_path).")

end

# ╔═╡ 9fa8c03a-48a7-4a26-94ed-bde694ac8f10
md"""
!!! tip
	Take note of the timestamped sub-directory so that you can load the saved model and parameters for use in the evaluation notebook.
"""

# ╔═╡ Cell order:
# ╟─f56c57c7-6301-4c07-a236-c11276eb1f32
# ╟─c43f57ce-90c5-4264-9e60-89f676bda905
# ╟─fefcdcfc-589e-482c-acc9-1e1a95f8d622
# ╟─f2ac80eb-f36d-4925-8ce4-c6a8dc8d42b6
# ╟─39ff3330-99b0-4cde-b9ba-9ec2d472dfda
# ╟─dc4d3b2d-f583-47f4-8368-52449829fb22
# ╟─8154b53a-09b9-4738-af9a-ffebf89a9e88
# ╠═05840afa-cc0e-442c-8869-0dac89e1f2c9
# ╟─912c1b67-8ba7-40f1-a4cf-33eaa4b6d7c6
# ╠═89124a6a-ecad-4570-86b0-f4e927e6fff0
# ╠═de110894-6faf-45e9-854d-1da663230472
# ╠═908065d1-6c48-4352-a34c-8d4d972799ff
# ╟─f8deace6-2bf9-49c5-98c0-1c18aca95aff
# ╟─359798d2-269b-4ae8-ae2d-e9fd3a26cf7e
# ╟─1072107e-e99b-46a9-ad10-9f6f894ad0da
# ╠═cd40a8a8-78f6-43f1-84c2-5664dbab402e
# ╟─5e56420c-a137-47b0-b2b7-e2c8c4341571
# ╟─9e4ad8a2-5a85-492e-bd9d-0f0f3f5e87d3
# ╠═e700c3f1-4ff1-46db-b6db-ad274f6ecb8a
# ╟─bc01f007-7500-4fed-afd6-47df14009859
# ╟─a4497815-598f-4ebe-a412-c3d145113f07
# ╠═b3d4bd35-213d-453b-9f7e-1d56f4202199
# ╟─d25d18cb-49e3-4c89-b729-07bcf5b33663
# ╟─cf012cc8-eda9-4487-8c38-8b8e80cc5984
# ╟─27dfaed4-a9d6-4c34-b25d-188607d53c20
# ╠═07d3c4ec-de35-40b0-b31c-ecaedaae622a
# ╟─da9d022e-dcde-42a4-af0a-a649dd3a744b
# ╟─da3367b9-80e7-4eba-b4e9-092541332645
# ╠═44f5b65a-41f7-4e2a-946c-942db95214a6
# ╠═1c7b43ea-9c2a-4596-9d2a-340f9731df63
# ╟─fa8bf210-437f-4bc9-81a1-31e30c10edb8
# ╟─cbffc631-9865-4b0c-a182-eb82c2420e90
# ╠═96c1c5fe-9abc-11ee-2316-4b1e5233a493
# ╟─9fa8c03a-48a7-4a26-94ed-bde694ac8f10
