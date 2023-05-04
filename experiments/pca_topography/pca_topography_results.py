import time
import auditory_cortex.analysis.analysis as analysis


strt = time.time()
pca_obj = analysis.PCA_topography()
sessions = pca_obj.get_significant_sessions()

l = 11

for session in sessions:
# session = 191113
    print(f"Computing results for session-{session}")
    # for layer in range(l,l+1):
    layer = l
    print(f"\t - Layer={layer}")
    for ch in range(pca_obj.get_all_channels(session).shape[0]):
    # ch = pca_obj.get_best_channel(session, layer)
        x, extent = pca_obj.get_kde(session, layer, ch)

end = time.time()

print(f"Took {(end - strt)/60.0} mins. to run..!")