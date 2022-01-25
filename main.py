import time
import fractions
import re

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

    

tokenizer = Tokenizer('plaintext.txt','cleantext.txt')
tokenizer.cleaner()
