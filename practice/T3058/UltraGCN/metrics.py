import numpy as np
'''
Test and metrics
'''

def hit(gt_item, pred_items):
	if gt_item in pred_items:
		return 1
	return 0


def ndcg(gt_item, pred_items):
	if gt_item in pred_items:
		index = pred_items.index(gt_item)
		return np.reciprocal(np.log2(index+2))
	return 0




def MRRatK_r(r, k):
	"""
    Mean Reciprocal Rank
    """
	pred_data = r[:, :k]
	scores = np.log2(1. / np.arange(1, k + 1))
	pred_data = pred_data / scores
	pred_data = pred_data.sum(1)
	return np.sum(pred_data)


def NDCGatK_r(test_data, r, k):
	"""
    Normalized Discounted Cumulative Gain
    rel_i = 1 or 0, so 2^{rel_i} - 1 = 1 or 0
    """
	assert len(r) == len(test_data)
	pred_data = r[:, :k]

	test_matrix = np.zeros((len(pred_data), k))
	for i, items in enumerate(test_data):
		length = k if k <= len(items) else len(items)
		test_matrix[i, :length] = 1
	max_r = test_matrix
	idcg = np.sum(max_r * 1. / np.log2(np.arange(2, k + 2)), axis=1)
	dcg = pred_data * (1. / np.log2(np.arange(2, k + 2)))
	dcg = np.sum(dcg, axis=1)
	idcg[idcg == 0.] = 1.
	ndcg = dcg / idcg
	ndcg[np.isnan(ndcg)] = 0.
	return np.sum(ndcg)
