import numpy

# Load data from hardcoded npy file
data = numpy.load("/Users/andrasjoos/Documents/AI_masters/Thesis/thesis_project/brain_age_pred/brain_gen/data/prior_stds_t1.npy")

# Save to hardcoded csv file
numpy.savetxt("/Users/andrasjoos/Documents/AI_masters/Thesis/thesis_project/brain_age_pred/brain_gen/data/prior_stds_t1.csv", data, delimiter=",") 

print(f"Successfully converted prior_means_t1.npy to prior_means_t1.csv")
print(f"Data shape: {data.shape}")