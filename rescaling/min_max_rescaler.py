""" quiz materials for feature scaling clustering """

### FYI, the most straightforward implementation might 
### throw a divide-by-zero error, if the min and max
### values are the same
### but think about this for a second--that means that every
### data point has the same value for that feature!  
### why would you rescale it?  Or even use it at all?
def featureScaling(arr):
    min_element = min(arr)
    max_element = max(arr)
    if min_element == max_element:
        # avoiding Division By Zero
        raise Exception("Cannot rescale values that have absolutely zero variance!")
    new_array = []
    for element in arr:
        rescaled = (float(element) - min_element) / (max_element - min_element)
        new_array.append(rescaled)
    return new_array

# tests of your feature scaler--line below is input data
data = [115, 140, 175]
print featureScaling(data)
