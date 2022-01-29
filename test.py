from msilib.schema import tables
import string

def generate_table(key):

        alphabet = "ABCDEFGHIKLMNOPQRSTUVWXYZ"

        table = []

        for char in key.upper():
            if char not in table and char in alphabet:
                table.append(char)
    

        for char in alphabet:
            if char not in table:
                table.append(char)
    
        return table



def prepare_input(dirty):

    
        dirty = "".join([c.upper() for c in dirty if c in string.ascii_letters])
        clean = ""
    
        if len(dirty) < 2:
            return dirty
    
        for i in range(len(dirty) - 1):
            clean += dirty[i]
    
            if dirty[i] == dirty[i + 1]:
                clean += "X"
    
        clean += dirty[-1]
    
        if len(clean) & 1:
            clean += "X"
    
        return clean

print(prepare_input('helloe'))