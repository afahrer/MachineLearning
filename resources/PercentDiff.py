def get(prediction, actual):
    if (prediction > actual):
        return (prediction - actual) / actual * 100
    elif (prediction < actual):
        return (actual - prediction) / actual * 100
    else:
        return 0
