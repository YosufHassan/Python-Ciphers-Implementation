from math import gcd
from main import Tokenizer


ALPHABET_SIZE = 26

class Affine():

    def encrypt_char(self,a, b, m, x):
        return (a*x+b)%m

    def decrypt_char(self,a, b, m, y):
        a_inv = self.inverse(a, m)
        return (a_inv * (y-b))%m

    def gcd(self,a, b):

        while b:
            a, b = b, a%b
        return a

    def inverse(self,x, m):
        possible_a_inv = [a for a in range(0,ALPHABET_SIZE) 
                            if self.gcd(a, ALPHABET_SIZE) == 1]
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
            x.append(self.mapDigitToAlpha(self.decrypt_char(a, b, m, self.mapAlphaToDigit(i))))

        return ''.join(x)

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
affine=Affine()
print(affine.inverse(7,180))