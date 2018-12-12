#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    cleaned_data = []

    tuples = []
    for index, net_worth_predicted in enumerate(predictions):
        net_worth_real = net_worths[index]
        age = ages[index]
        residual_error = abs(net_worth_predicted - net_worth_real)
        tuples.append((age, net_worth_real, residual_error))
        
    tuples.sort(key=lambda tup:tup[2], reverse=True)
    
    cleaned_data = tuples[9:]
    
    return cleaned_data

