from cbir.retrieval import retrieve_similar_images

query_image = r"dataset\daun_salam\foto depan\daun salam 1.jpeg"

results = retrieve_similar_images(
    query_image,
    top_k=5
)

for i, result in enumerate(results, start=1):

    print(
        f"{i}. {result['path']} "
        f"Distance: {result['distance']:.4f}"
    )