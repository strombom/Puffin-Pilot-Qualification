

class GateFinder:
	def __init__(self):
		pass

	def find_gates(self, image):

		n_boxes = 1
        bb_all = 400*np.random.uniform(size = (n_boxes,9))
        bb_all[:,-1] = 0.5

        return bb_all.tolist()
