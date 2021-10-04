def sort_dic(results):
    sorted_values = sorted(results.values()) # Sort the values
    sorted_values.reverse()
    sorted_dict = {}

    for i in sorted_values:
        for k in results.keys():
            if results[k] == i:
                sorted_dict[k] = abs(results[k])
    return sorted_dict

def rename_dict(sorted_dict):
    #Renames dict to latex

    temp = {}
    for x in sorted_dict.keys():
        #print(x)
        #Translate Name to Latex
        name = '${'
        if('AE' in x):
            name += '\widehat{'
        elif('BE' in x):
            name += '{'
        if('QS' in x):
            name += 'Q}_{SQ}'
        elif('QE' in x):
            name += 'Q}_{E}'
        elif('fro' in x):
            name += 'Q}_{F}'
        elif('spec' in x):
            name += 'Q}_{S}'
        if('L1' in x):
            name += '^{L1}'
        elif('L2' in x):
            name += '^{L2}'
        elif('L3' in x):
            name += '^{p}'
        elif('L4' in x):
            name += '^{L4}'
        elif('L5' in x):
            name += '^{L5}'

        name += '}$'
        name += "-"+(x.split("-")[-1]).upper()
        #print(name)
        temp[name] = sorted_dict[x]

    return temp

