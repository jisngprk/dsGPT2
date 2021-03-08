

def conv_filter(data):
    if 'q' in data:
        data['q'] = str(data['q'])
    if 'a' in data:
        data['a'] = str(data['a'])

    return data
