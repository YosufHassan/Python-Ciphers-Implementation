from lib2to3.pgen2 import token
import time
import fractions
import re
import string
import itertools
import math
from tokenize import Token

key = "HACK"
ALPHABET_SIZE = 26

class Tokenizer():

    def __init__(self, plaintext, cleantext):
        self.plaintext,self.cleantext = plaintext,cleantext

    def cleaner(self):
        uncleanText = open(self.plaintext).read()
        cleanText = re.sub('[^A-Za-z0-9\s\n]+', '', uncleanText)
        open(self.cleantext, 'w').write(cleanText)

    def readFile(self):
        fileObj = open(self.cleantext, "r")
        tokens = fileObj.read().split()
        fileObj.close()
        return tokens

    def tokenSplitter(token):
        return [char for char in token]

    def tokenArrSplitter(self, arr):
        clean = []
        for token in range(len(arr)-1):
            clean.append(Tokenizer.tokenSplitter(arr[token]))
        return clean
    
    def concatList(self,list):
        reslist = " ".join(list)
        return reslist

class Affine():

    def mapAlphaToDigit(self,x):
        if str.isdigit(x):
            i = ord(x)
            if 47 < i and i < 58:
                return ord(x)-22
        elif str.isalpha(x):
            return ord(x)-97

    def mapDigitToAlpha(self,x):
        if 0 <= x and x < 26:
            return chr(x+97)
        elif 26 <= x and x < ALPHABET_SIZE:
            return chr(x+22)

    def encrypt_char(self,a, b, m, x):
        return (a*x+b)%m

    def decrypt_char(self,a, b, m, y):
        a_inv = self.inverse(a, m)
        return (a_inv * (y-b))%m

    def inverse(x, m):
        possible_a_inv = [a for a in range(0,ALPHABET_SIZE) 
                            if fractions.gcd(a, ALPHABET_SIZE) == 1]
        for i in possible_a_inv:
            if (x*i)%m == 1:
                return i

    def encrypt(self,a, b, x, m):
        y = []
        for i in x:
            y.append(self.mapDigitToAlpha(self.encrypt_char(a, b, m, self.mapAlphaToDigit(i))))

        return ''.join(y)

    def decrypt(self,a, b, y, m):
        x = []
        for i in y:
            x.append(self.mapDigitToAlpha(self,self.decrypt_char(a, b, m, self.mapAlphaToDigit(i))))

        return ''.join(x)



class PlayFair():
    def chunker(seq, size):
        it = iter(seq)
        while True:
            chunk = tuple(itertools.islice(it, size))
            if not chunk:
                return
            yield chunk
    
    
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
    
    
    def encode(self,plaintext, key):
        table = self.generate_table(key)
        plaintext = self.prepare_input(plaintext)
        ciphertext = ""

        for char1, char2 in self.chunker(plaintext, 2):
            row1, col1 = divmod(table.index(char1), 5)
            row2, col2 = divmod(table.index(char2), 5)
    
            if row1 == row2:
                ciphertext += table[row1 * 5 + (col1 + 1) % 5]
                ciphertext += table[row2 * 5 + (col2 + 1) % 5]
            elif col1 == col2:
                ciphertext += table[((row1 + 1) % 5) * 5 + col1]
                ciphertext += table[((row2 + 1) % 5) * 5 + col2]
            else: 
                ciphertext += table[row1 * 5 + col2]
                ciphertext += table[row2 * 5 + col1]
    
        return ciphertext
    
    
    def decode(self,ciphertext, key):
        table = self.generate_table(key)
        plaintext = ""
    
        for char1, char2 in self.chunker(ciphertext, 2):
            row1, col1 = divmod(table.index(char1), 5)
            row2, col2 = divmod(table.index(char2), 5)
    
            if row1 == row2:
                plaintext += table[row1 * 5 + (col1 - 1) % 5]
                plaintext += table[row2 * 5 + (col2 - 1) % 5]
            elif col1 == col2:
                plaintext += table[((row1 - 1) % 5) * 5 + col1]
                plaintext += table[((row2 - 1) % 5) * 5 + col2]
            else:
                plaintext += table[row1 * 5 + col2]
                plaintext += table[row2 * 5 + col1]
    
        return plaintext

class ColumnTransposition():
# Encryption
    def encryptMessage(msg):
        cipher = ""

        # track key indices
        k_indx = 0

        msg_len = float(len(msg))
        msg_lst = list(msg)
        key_lst = sorted(list(key))

        # calculate column of the matrix
        col = len(key)
        
        # calculate maximum row of the matrix
        row = int(math.ceil(msg_len / col))

        # add the padding character '_' in empty
        # the empty cell of the matix
        fill_null = int((row * col) - msg_len)
        msg_lst.extend('_' * fill_null)

        # create Matrix and insert message and
        # padding characters row-wise
        matrix = [msg_lst[i: i + col]
                for i in range(0, len(msg_lst), col)]

        # read matrix column-wise using key
        for _ in range(col):
            curr_idx = key.index(key_lst[k_indx])
            cipher += ''.join([row[curr_idx]
                            for row in matrix])
            k_indx += 1

        return cipher

    # Decryption
    def decryptMessage(cipher):
        msg = ""

        # track key indices
        k_indx = 0

        # track msg indices
        msg_indx = 0
        msg_len = float(len(cipher))
        msg_lst = list(cipher)

        # calculate column of the matrix
        col = len(key)
        
        # calculate maximum row of the matrix
        row = int(math.ceil(msg_len / col))

        # convert key into list and sort
        # alphabetically so we can access
        # each character by its alphabetical position.
        key_lst = sorted(list(key))

        # create an empty matrix to
        # store deciphered message
        dec_cipher = []
        for _ in range(row):
            dec_cipher += [[None] * col]

        # Arrange the matrix column wise according
        # to permutation order by adding into new matrix
        for _ in range(col):
            curr_idx = key.index(key_lst[k_indx])

            for j in range(row):
                dec_cipher[j][curr_idx] = msg_lst[msg_indx]
                msg_indx += 1
            k_indx += 1

        # convert decrypted msg matrix into a string
        try:
            msg = ''.join(sum(dec_cipher, []))
        except TypeError:
            raise TypeError("This program cannot",
                            "handle repeating words.")

        null_count = msg.count('_')

        if null_count > 0:
            return msg[: -null_count]

        return msg

# Driver code for classical algorithms
print('This is a benchmark to time classical algorithms\n')
print('Input file: "plaintext.txt"\n')
tokenizer = Tokenizer('plaintext.txt','cleantext.txt')
print("Cleaning input file ...\n")
tokenizer.cleaner()
print("parsing cleaned input file ...\n\nThe message is as follows\n")
tokens = tokenizer.readFile()
print(tokenizer.concatList(tokens))
print('\nPerforming the Affine encryption algorithm ...\n\nThe encrypted message is as follows\n')
affineres = []
affinestart = time.time()
for i in range(len(tokens)-1):
    affineres.append(Affine().encrypt(17,25,tokens[i],ALPHABET_SIZE))
affineend = time.time()
print(tokenizer.concatList(affineres))
print(f"\nTime taken to encrypt the file is {affineend-affinestart} seconds\n\n")