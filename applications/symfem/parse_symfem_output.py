import re 

def process_output(output):
    
    # add line break after every comma
    output = output.replace(",",",\n")
    
    # replace with array entries
    output = re.sub(r'c(\d)(\d)',
                    lambda m: f'c[{int(m.group(1))-1},{int(m.group(2))-1}]',
                    output)
    
    return output

if __name__ == "__main__":
    file = "symfem_output.txt"
    with open(file,"r") as f:
        output = f.read()
    print(process_output(output))
        