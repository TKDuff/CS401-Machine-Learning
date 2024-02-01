### A Pluto.jl notebook ###
# v0.19.35

using Markdown
using InteractiveUtils

# ╔═╡ 00b74233-615c-464a-bf06-ceaff6891882
begin
	import Pkg
    Pkg.activate(mktempdir())
	Pkg.develop(url="https://github.com/rgreilly/Transformers")    
	Pkg.add(["Revise", "PlutoUI", "Flux", "Plots", "DataFrames", "Printf", 
		"BSON", "JSON", "Arrow", "StatsBase", "Unicode", "Random", 
		"DataStructures", "ProgressMeter", "RemoteFiles"])
	
	using Revise
	using TransformersLite
	using PlutoUI
	using Flux
	using Flux: DataLoader
	using Plots
	using DataFrames
	using Printf
	using BSON, JSON
	using Arrow
	using StatsBase
	using Unicode
	using Random
	using DataStructures
	using ProgressMeter
	using RemoteFiles
end;

# ╔═╡ 766e5d43-f09f-460e-b812-51eb0678cc89
md"""
![transformer](https://github.com/rgreilly/images/assets/1801654/f5d27ba2-5f19-4b53-8cdf-17defc3a910b)
"""

# ╔═╡ 2ebe0309-e918-41a1-831b-ba9c0d5faffa
md"""
# Sentiment analysis using a transformer
## Model evaluation
"""

# ╔═╡ 2e40fe64-14f2-421e-8665-9ae45642dd37
TableOfContents()

# ╔═╡ c2e388b6-38db-4a80-920c-92ac343b36f9
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

# ╔═╡ 30aab056-83ec-4124-9ae9-f50b71d960dc
begin
	@RemoteFileSet FILES "Transformer utilities" begin
	    reporting = @RemoteFile "https://github.com/rgreilly/Transformers/blob/main/examples/reporting.jl" dir="utilities" file="reporting.jl.json"
	    utilities = @RemoteFile "https://github.com/rgreilly/Transformers/blob/main/examples/utilities.jl" dir="utilities" file="utilities.jl.json"
		training = @RemoteFile "https://github.com/rgreilly/Transformers/blob/main/examples/training.jl" dir="utilities" file="training.jl.json"
	end

	download(FILES) # Files downloaded in JSON format
end

# ╔═╡ 86ea40f3-8399-4ded-b811-cfbc70d98604
function convertJSON(inFile, outFile)
	body = JSON.parsefile(inFile)["payload"]["blob"]["rawLines"]
	open(outFile, "w") do f
	  for i in body
	    println(f, i)
	  end
	end 
end

# ╔═╡ d89c4def-7052-424f-a26b-e458f8ad859c
begin
	convertJSON("utilities/reporting.jl.json", "utilities/reporting.jl")
	convertJSON("utilities/utilities.jl.json", "utilities/utilities.jl")
	convertJSON("utilities/training.jl.json", "utilities/training.jl")
	
	include("utilities/reporting.jl")
	include("utilities/utilities.jl")
	include("utilities/training.jl")
end;

# ╔═╡ aef1e7f8-3dee-4c89-b790-928de9d4e8d7
md"""
Multi-class classification: stars from 1 to 5
"""

# ╔═╡ c2336c86-96c1-4d28-92dc-f49b05cbf7da
md"""
## Load data
"""

# ╔═╡ a7d5db4c-f010-48e6-a9ba-6f4d24ef0fa9
md"""
The original CSV data format has been converted to arrow format for faster loading.
"""

# ╔═╡ df13c372-d23e-427e-b4fb-88238bb69f15
begin
	path = "datasets/amazon_reviews_multi/en/1.0.0/"
	file_train = "train.arrow"
	file_test = "test.arrow" 
	nlabels = 5
end;

# ╔═╡ 56d8b8ad-c038-4272-b1dd-12d723482bed
md"""
Load training and test data into dataframes and about the size of both. 
"""

# ╔═╡ aaff9c59-9149-472a-ad38-e75a62486eb8
begin
	filepath = joinpath(path, file_train)
	df = DataFrame(Arrow.Table(filepath))
	
	filepath = joinpath(path, file_test)
	df_test = DataFrame(Arrow.Table(filepath))

	(nrow(df), nrow(df_test))
end

# ╔═╡ 38cfa3d7-bc3d-4ee1-b775-5e7fb0f2b405
md"""
Extract just the review text and the star rating
"""

# ╔═╡ 2d55eaa7-cd9d-49d9-a0c7-08880c173da8
begin
	documents = df[:, "review_body"]
	labels = df[:, "stars"]
	
	println("training samples: ", size(documents), " ", size(labels))
end

# ╔═╡ 0cd4cc0d-7ba0-4bc6-aba9-bdef4f92b462
md"""
Do the same for the test data
"""

# ╔═╡ 6d7ca181-0e83-4d9b-bfd0-9296e3ad97a9
begin
	documents_test = df_test[:, "review_body"]
	labels_test = df_test[:, "stars"];
	
	println("test samples: ", size(documents_test), " ", size(labels_test))
end

# ╔═╡ 1c0ff187-6fa2-4215-8465-6921fc7908a3
md"""
Load the already trained and saved model.  Note that models and associated details are stored in the `outputs` directory under a sub-directory name generated from the time it was saved in the format: `yyyymmdd_hhmm`.
"""

# ╔═╡ 47d2c5d6-b372-458e-8f66-fc0cad6a8842
begin
	directory = "../outputs/20231217_2146/"
	saved_objects = BSON.load(joinpath(directory, "model.bson"))
	tokenizer = saved_objects[:tokenizer]
	@show tokenizer
	indexer = saved_objects[:indexer]
	@show indexer
	model = saved_objects[:model]
	display(model)
end;

# ╔═╡ 812055c2-0318-4306-a9ab-491701edba13
md"""
## Tokenise
"""

# ╔═╡ a37997c0-5816-490d-9d0c-433e834efc54
md"""
Tokenise the training and test data
"""

# ╔═╡ ea7f7092-c55e-468d-8c60-f0e7ad980b71
begin
	max_length = size(model.classifier.weight, 2)
	@time tokens = map(d->preprocess(d, tokenizer, max_length=max_length), documents) #takes about 30 seconds for all documents
	@time indices = indexer(tokens) #takes about 12 seconds for all documents
	
	y_train = copy(labels)
	idxs = Base.OneTo(length(labels))
	X_train, y_train = indices[:, idxs], y_train[idxs];
	y_train = Flux.onehotbatch(y_train, 1:5) # multi-class
	train_data, val_data = split_validation(X_train, y_train; 
		rng=MersenneTwister(2718))
	
	println("train samples:      ", size(train_data[1]), " ", size(train_data[2]))
	println("validation samples: ", size(val_data[1]), " ", size(val_data[2]))
end

# ╔═╡ 7d7ab4b9-053d-4157-80e4-04845e7ad522
begin
	y_test = copy(labels_test)
	y_test = Flux.onehotbatch(y_test, 1:5);
	
	@time tokens_test = map(d->preprocess(d, tokenizer, max_length=max_length), documents_test) 
	@time indices_test = indexer(tokens_test)
	
	X_test = indices_test
	
	println("test indices: ", size(indices_test))
	println("test samples: ", size(X_test), " ", size(y_test))
end

# ╔═╡ 898edd10-3b9e-4acc-8109-1fa44afe344c
md"""
Create the training and validation data loaders
"""

# ╔═╡ f0b308f2-892e-4bf7-b5c2-afac0db0b043
begin
	train_data_loader = DataLoader(train_data; batchsize=64, shuffle=false);
	val_data_loader  = DataLoader(val_data; batchsize=64, shuffle=false);
end

# ╔═╡ c66ebdee-15a0-4014-a4cc-40d5112fc940
md"""
## Evaluate
"""

# ╔═╡ 2513c6f0-629b-4c79-973f-acaea2977ace
begin
	loss(x, y) = Flux.logitcrossentropy(model(x), y)
	loss(x::Tuple) = loss(x[1], x[2])
	accuracy(ŷ, y) = mean(Flux.onecold(ŷ) .== Flux.onecold(y))
end

# ╔═╡ 583e12f7-21d8-46c7-9fc8-179996b1ee4d
@time batched_metric(model, accuracy, train_data_loader)

# ╔═╡ 99747375-96b9-408a-8915-6c1ad4f4bf21
@time batched_metric(model, accuracy, val_data_loader)

# ╔═╡ 4e6cda11-7ea9-49f9-98c0-76c62a7d07ba
history = open(joinpath(directory, "history.json"), "r") do f
    JSON.parse(read(f, String))
end

# ╔═╡ e7227497-a8d4-424f-b510-8ceea9ac3b13
begin
	epochs = 1:length(history["train_acc"])
	p1 = plot(epochs, history["train_acc"], label="train")
	plot!(p1, epochs, history["val_acc"], label="val")
	plot!(p1, ylims=[0, 1], title="accuracy", legend=(0.9, 0.8))
	
	p2 = plot(epochs, history["train_loss"], label="train")
	plot!(p2, epochs, history["val_loss"], label="val")
	plot!(p2, title="loss", ylims=[0, Inf], legend=(0.8, 0.5))
	
	p3 = plot(p1, p2, layout=grid(1, 2), size=(800, 300))
	savefig(p3, joinpath(directory, "history.png"))
	p3
end

# ╔═╡ bbc3f501-d8cc-44af-9f9c-3e8f309c4c6a
md"""
## Test data
"""

# ╔═╡ 5d6dea4a-1c13-4982-bc75-a719a8e8ff05
begin
	logits = model(X_test)
	accuracy(logits, y_test)
end

# ╔═╡ 15618c1c-9129-4aea-ba6d-1dc45f9a01fc
begin
	probs = softmax(logits, dims=1)
	y_pred = Flux.onecold(probs);
end

# ╔═╡ 75b31344-44b9-4c41-8027-4394509e1661
cm = confusion_matrix(vec(y_pred), Flux.onecold(y_test), 1:nlabels)

# ╔═╡ 1d2afc0e-3b66-4930-805a-b5501da187de
begin
	p4 = heatmap(1:5, 1:5, cm, xlabel="predictions", ylabel="ground truth", xlims=(0.5, nlabels+0.5), aspectratio=1,
	    title="confusion matrix test samples", xticks=(1:5)) #, ["negative", "mix", "positive"]))
	savefig(p4, joinpath(directory, "confusion_matrix.png"))
	p4
end

# ╔═╡ b3864f5c-32bf-445a-be8b-c1db6960312e
classification_report(cm, 1:nlabels)

# ╔═╡ 88359448-f5d9-4947-b185-3742ecac9aa7
md"""
### Examples
"""

# ╔═╡ d6257030-2517-468c-bc4c-086c044cc832
begin
	println("star  y  ŷ   prob")
	for star in nlabels:-1:1
	    pos_max = argmax(probs[star, :])
	    @printf("   %1d  %d  %d  %.4f\n %s\n\n",
	            star, labels_test[pos_max], y_pred[pos_max], probs[star, pos_max], documents_test[pos_max]
	        )
	end
end

# ╔═╡ 2ded6eab-8c7c-457d-ba2d-42e1b45ab290
md"""
### Probabilities
"""

# ╔═╡ 520e84cb-a12e-4629-8473-e52225cb67e5
begin
	canvases1 = []
	label_names = 1:5
	for gt_star in 1:5
	    idxs = labels_test .== gt_star
	    value_counts = [sum((y_pred[idxs]) .== l) for l in 1:nlabels]
	    p = bar(value_counts, xlabel="star=$gt_star",legend=:none, xticks=(1:nlabels, 1:5))#["neg", "mix", "pos"]))
	    push!(canvases1, p)
	end
	plot!(canvases1[1], ylabel="frequency")
	p5 =plot(canvases1..., layout=(1, 5), link=:y, size=(900, 400), plot_title="predicted class per ground truth class",
	    margin=5Plots.mm)
	savefig(p5, joinpath(directory, "prediction_star.png"))
	p5
end

# ╔═╡ 250deb49-b410-4659-b579-d9fcbe14c626
md"""
### Single sample
"""

# ╔═╡ df45281f-31cf-4f45-95d8-c7f7428bf26c
begin
	idx = 4600 
	
	d = documents_test[idx]
	println(labels_test[idx])
	println(d)
	println("")
	
	tokens2 = preprocess(d, tokenizer, max_length=50)
	println(join(tokens2, "|"))
	println("")
	
	x = indexer(tokens2)
	x = vcat(x, ones(Int, 50 - length(x)))
	println(join(x, "|"))
end

# ╔═╡ 7ae8a8ae-2bb1-4729-bd6f-7c085c7d477b
softmax(model(x))

# ╔═╡ Cell order:
# ╟─766e5d43-f09f-460e-b812-51eb0678cc89
# ╟─2ebe0309-e918-41a1-831b-ba9c0d5faffa
# ╟─2e40fe64-14f2-421e-8665-9ae45642dd37
# ╟─c2e388b6-38db-4a80-920c-92ac343b36f9
# ╠═00b74233-615c-464a-bf06-ceaff6891882
# ╠═30aab056-83ec-4124-9ae9-f50b71d960dc
# ╠═86ea40f3-8399-4ded-b811-cfbc70d98604
# ╠═d89c4def-7052-424f-a26b-e458f8ad859c
# ╟─aef1e7f8-3dee-4c89-b790-928de9d4e8d7
# ╟─c2336c86-96c1-4d28-92dc-f49b05cbf7da
# ╟─a7d5db4c-f010-48e6-a9ba-6f4d24ef0fa9
# ╠═df13c372-d23e-427e-b4fb-88238bb69f15
# ╟─56d8b8ad-c038-4272-b1dd-12d723482bed
# ╠═aaff9c59-9149-472a-ad38-e75a62486eb8
# ╟─38cfa3d7-bc3d-4ee1-b775-5e7fb0f2b405
# ╠═2d55eaa7-cd9d-49d9-a0c7-08880c173da8
# ╟─0cd4cc0d-7ba0-4bc6-aba9-bdef4f92b462
# ╠═6d7ca181-0e83-4d9b-bfd0-9296e3ad97a9
# ╟─1c0ff187-6fa2-4215-8465-6921fc7908a3
# ╠═47d2c5d6-b372-458e-8f66-fc0cad6a8842
# ╟─812055c2-0318-4306-a9ab-491701edba13
# ╟─a37997c0-5816-490d-9d0c-433e834efc54
# ╠═ea7f7092-c55e-468d-8c60-f0e7ad980b71
# ╠═7d7ab4b9-053d-4157-80e4-04845e7ad522
# ╟─898edd10-3b9e-4acc-8109-1fa44afe344c
# ╠═f0b308f2-892e-4bf7-b5c2-afac0db0b043
# ╟─c66ebdee-15a0-4014-a4cc-40d5112fc940
# ╠═2513c6f0-629b-4c79-973f-acaea2977ace
# ╠═583e12f7-21d8-46c7-9fc8-179996b1ee4d
# ╠═99747375-96b9-408a-8915-6c1ad4f4bf21
# ╠═4e6cda11-7ea9-49f9-98c0-76c62a7d07ba
# ╠═e7227497-a8d4-424f-b510-8ceea9ac3b13
# ╟─bbc3f501-d8cc-44af-9f9c-3e8f309c4c6a
# ╠═5d6dea4a-1c13-4982-bc75-a719a8e8ff05
# ╠═15618c1c-9129-4aea-ba6d-1dc45f9a01fc
# ╠═75b31344-44b9-4c41-8027-4394509e1661
# ╠═1d2afc0e-3b66-4930-805a-b5501da187de
# ╠═b3864f5c-32bf-445a-be8b-c1db6960312e
# ╟─88359448-f5d9-4947-b185-3742ecac9aa7
# ╠═d6257030-2517-468c-bc4c-086c044cc832
# ╟─2ded6eab-8c7c-457d-ba2d-42e1b45ab290
# ╠═520e84cb-a12e-4629-8473-e52225cb67e5
# ╟─250deb49-b410-4659-b579-d9fcbe14c626
# ╠═df45281f-31cf-4f45-95d8-c7f7428bf26c
# ╠═7ae8a8ae-2bb1-4729-bd6f-7c085c7d477b
