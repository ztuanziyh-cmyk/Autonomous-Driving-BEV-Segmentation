import numpy as np
import utils
palette = utils.parse_convert_xml("one_hot_conversion/convert_3+occl.xml")
dist = utils.get_class_distribution("../data/2_F/train/bev+occlusion", (256, 512), palette)
weights = np.log(np.reciprocal(list(dist.values())))
print(weights)
