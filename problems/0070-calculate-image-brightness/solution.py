
def calculate_brightness(img):
	# Write your code here
	if img == []:
        return -1
    else:
        lengths = []
    
        for i in range(len(img)):
            if max(img[i]) > 255 or min(img[i]) < 0:
                return -1
            else:
                lengths.append(len(img[i]))
        if max(lengths) != min(lengths):
            return -1
        else:
            def flatten(xss):
                return [x for xs in xss for x in xs]
            flat_list = flatten(img)
            return sum(flat_list) / len(flat_list)

