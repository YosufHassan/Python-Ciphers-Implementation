import time
import re
import string
import itertools
import math
import os

key = "qvgcwxiybjspzfadtelnkuorhm"  # Key used for column transposition and playfair algorithm
ALPHABET_SIZE = 26

# Global Variables and helper functions for AES

# The SBox given in the design
s_box = (
    0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
    0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
    0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
    0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
    0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
    0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
    0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
    0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
    0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
    0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
    0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
    0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
    0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
    0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
    0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
    0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16,
)

# The  inverse of the SBox
inv_s_box = (
    0x52, 0x09, 0x6A, 0xD5, 0x30, 0x36, 0xA5, 0x38, 0xBF, 0x40, 0xA3, 0x9E, 0x81, 0xF3, 0xD7, 0xFB,
    0x7C, 0xE3, 0x39, 0x82, 0x9B, 0x2F, 0xFF, 0x87, 0x34, 0x8E, 0x43, 0x44, 0xC4, 0xDE, 0xE9, 0xCB,
    0x54, 0x7B, 0x94, 0x32, 0xA6, 0xC2, 0x23, 0x3D, 0xEE, 0x4C, 0x95, 0x0B, 0x42, 0xFA, 0xC3, 0x4E,
    0x08, 0x2E, 0xA1, 0x66, 0x28, 0xD9, 0x24, 0xB2, 0x76, 0x5B, 0xA2, 0x49, 0x6D, 0x8B, 0xD1, 0x25,
    0x72, 0xF8, 0xF6, 0x64, 0x86, 0x68, 0x98, 0x16, 0xD4, 0xA4, 0x5C, 0xCC, 0x5D, 0x65, 0xB6, 0x92,
    0x6C, 0x70, 0x48, 0x50, 0xFD, 0xED, 0xB9, 0xDA, 0x5E, 0x15, 0x46, 0x57, 0xA7, 0x8D, 0x9D, 0x84,
    0x90, 0xD8, 0xAB, 0x00, 0x8C, 0xBC, 0xD3, 0x0A, 0xF7, 0xE4, 0x58, 0x05, 0xB8, 0xB3, 0x45, 0x06,
    0xD0, 0x2C, 0x1E, 0x8F, 0xCA, 0x3F, 0x0F, 0x02, 0xC1, 0xAF, 0xBD, 0x03, 0x01, 0x13, 0x8A, 0x6B,
    0x3A, 0x91, 0x11, 0x41, 0x4F, 0x67, 0xDC, 0xEA, 0x97, 0xF2, 0xCF, 0xCE, 0xF0, 0xB4, 0xE6, 0x73,
    0x96, 0xAC, 0x74, 0x22, 0xE7, 0xAD, 0x35, 0x85, 0xE2, 0xF9, 0x37, 0xE8, 0x1C, 0x75, 0xDF, 0x6E,
    0x47, 0xF1, 0x1A, 0x71, 0x1D, 0x29, 0xC5, 0x89, 0x6F, 0xB7, 0x62, 0x0E, 0xAA, 0x18, 0xBE, 0x1B,
    0xFC, 0x56, 0x3E, 0x4B, 0xC6, 0xD2, 0x79, 0x20, 0x9A, 0xDB, 0xC0, 0xFE, 0x78, 0xCD, 0x5A, 0xF4,
    0x1F, 0xDD, 0xA8, 0x33, 0x88, 0x07, 0xC7, 0x31, 0xB1, 0x12, 0x10, 0x59, 0x27, 0x80, 0xEC, 0x5F,
    0x60, 0x51, 0x7F, 0xA9, 0x19, 0xB5, 0x4A, 0x0D, 0x2D, 0xE5, 0x7A, 0x9F, 0x93, 0xC9, 0x9C, 0xEF,
    0xA0, 0xE0, 0x3B, 0x4D, 0xAE, 0x2A, 0xF5, 0xB0, 0xC8, 0xEB, 0xBB, 0x3C, 0x83, 0x53, 0x99, 0x61,
    0x17, 0x2B, 0x04, 0x7E, 0xBA, 0x77, 0xD6, 0x26, 0xE1, 0x69, 0x14, 0x63, 0x55, 0x21, 0x0C, 0x7D,
)

# Function that performs the subsitution of the bytes for the AES algorithm
def sub_bytes(s):
    for i in range(4):
        for j in range(4):
            s[i][j] = s_box[s[i][j]]
    print(f"\nSubBytes for current Block\n\n{s}")

def inv_sub_bytes(s):
    for i in range(4):
        for j in range(4):
            s[i][j] = inv_s_box[s[i][j]]

# Function that is responsible to shift the rows for the AES algorithm encryption
def shift_rows(s):
    s[0][1], s[1][1], s[2][1], s[3][1] = s[1][1], s[2][1], s[3][1], s[0][1]
    s[0][2], s[1][2], s[2][2], s[3][2] = s[2][2], s[3][2], s[0][2], s[1][2]
    s[0][3], s[1][3], s[2][3], s[3][3] = s[3][3], s[0][3], s[1][3], s[2][3]
    print(f"\nShift rows for current Block\n\n{s}\n")
    
def inv_shift_rows(s):
    s[0][1], s[1][1], s[2][1], s[3][1] = s[3][1], s[0][1], s[1][1], s[2][1]
    s[0][2], s[1][2], s[2][2], s[3][2] = s[2][2], s[3][2], s[0][2], s[1][2]
    s[0][3], s[1][3], s[2][3], s[3][3] = s[1][3], s[2][3], s[3][3], s[0][3]

# Function that is responsible to reverse the shifting of the rows for the AES algorithm decryption
def add_round_key(s, k):
    for i in range(4):
        for j in range(4):
            s[i][j] ^= k[i][j]
    print(f'\nadd round key for current block\n\n{s}\n')


xtime = lambda a: (((a << 1) ^ 0x1B) & 0xFF) if (a & 0x80) else (a << 1)


def mix_single_column(a):

    t = a[0] ^ a[1] ^ a[2] ^ a[3]
    u = a[0]
    a[0] ^= t ^ xtime(a[0] ^ a[1])
    a[1] ^= t ^ xtime(a[1] ^ a[2])
    a[2] ^= t ^ xtime(a[2] ^ a[3])
    a[3] ^= t ^ xtime(a[3] ^ u)


def mix_columns(s):
    for i in range(4):
        mix_single_column(s[i])

    print(f'\nMix Columns for current block\n\n{s}\n')

def inv_mix_columns(s):

    for i in range(4):
        u = xtime(xtime(s[i][0] ^ s[i][2]))
        v = xtime(xtime(s[i][1] ^ s[i][3]))
        s[i][0] ^= u
        s[i][1] ^= v
        s[i][2] ^= u
        s[i][3] ^= v

    mix_columns(s)


r_con = (
    0x00, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40,
    0x80, 0x1B, 0x36, 0x6C, 0xD8, 0xAB, 0x4D, 0x9A,
    0x2F, 0x5E, 0xBC, 0x63, 0xC6, 0x97, 0x35, 0x6A,
    0xD4, 0xB3, 0x7D, 0xFA, 0xEF, 0xC5, 0x91, 0x39,
)


def bytes2matrix(text):
    """ Converts a 16-byte array into a 4x4 matrix.  """
    return [list(text[i:i+4]) for i in range(0, len(text), 4)]

def matrix2bytes(matrix):
    """ Converts a 4x4 matrix into a 16-byte array.  """
    return bytes(sum(matrix, []))

def xor_bytes(a, b):
    """ Returns a new byte array with the elements xor'ed. """
    return bytes(i^j for i, j in zip(a, b))

def inc_bytes(a):
    """ Returns a new byte array with the value increment by 1 """
    out = list(a)
    for i in reversed(range(len(out))):
        if out[i] == 0xFF:
            out[i] = 0
        else:
            out[i] += 1
            break
    return bytes(out)

def pad(plaintext):
    """
    Pads the given plaintext with PKCS#7 padding to a multiple of 16 bytes.
    Note that if the plaintext size is a multiple of 16,
    a whole block will be added.
    """
    padding_len = 16 - (len(plaintext) % 16)
    padding = bytes([padding_len] * padding_len)
    return plaintext + padding

def unpad(plaintext):
    """
    Removes a PKCS#7 padding, returning the unpadded text and ensuring the
    padding was correct.
    """
    padding_len = plaintext[-1]
    assert padding_len > 0
    message, padding = plaintext[:-padding_len], plaintext[-padding_len:]
    assert all(p == padding_len for p in padding)
    return message

def split_blocks(message, block_size=16, require_padding=True):
        assert len(message) % block_size == 0 or not require_padding
        return [message[i:i+16] for i in range(0, len(message), block_size)]

# Class responsible for reading and parsing the input file in addition the class provides useful operations to manipulate the file contents and format user output
class Tokenizer():

    def __init__(self, plaintext, cleantext):
        self.plaintext,self.cleantext = plaintext,cleantext

    def cleaner(self):
        uncleanText = open(self.plaintext).read()
        cleanText = re.sub('[^A-Za-z0-9\s\n]+', '', uncleanText)
        open(self.cleantext, 'w').write(cleanText.lower())

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

class PlayFair():
    def chunker(self,seq, size): # Split the plaintext into pairs and return the result as list of pairs
        it = iter(seq)
        while True:
            chunk = tuple(itertools.islice(it, size))
            if not chunk:
                return
            yield chunk
    
    
    def prepare_input(self,dirty):

        # Capitalize plaintext and include all the letters and remove anything that is not in the ascii table
        dirty = "".join([c.upper() for c in dirty if c in string.ascii_letters])
        clean = ""
    
        if len(dirty) < 2:
            return dirty
    
        for i in range(len(dirty) - 1): # Add the junk letter X if the pair has the same letter
            clean += dirty[i]
    
            if dirty[i] == dirty[i + 1]:
                clean += "X"
    
        clean += dirty[-1]            
    
        if len(clean) & 1:
            clean += "X"            # Add the junk letter X if the number of letters is not even
    
        return clean
    
    
    def generate_table(self,key):

        alphabet = "ABCDEFGHIKLMNOPQRSTUVWXYZ" # Define List of alphabets

        table = [] # Define key matrix

        for char in key.upper():
            if char not in table and char in alphabet:  # check if the letters in the encryption key are in the table or not
                table.append(char)                      # if some of the letters are not in the key matrix they are appended
    

        for char in alphabet:                           # check if the letters in the alphabets are in the table or not
            if char not in table:
                table.append(char)                      # if some of the letters are not in the key matrix they are appended
    
        return table
    
    
    def encode(self,plaintext, key):
        table = self.generate_table(key)
        print(f'Constructing square key matrix\n\n{table}\n')
        plaintext = self.prepare_input(plaintext)
        ciphertext = ""

        for char1, char2 in self.chunker(plaintext, 2): # Each pair identify which row and column the pair exists (identify pair position)
            row1, col1 = divmod(table.index(char1), 5)
            row2, col2 = divmod(table.index(char2), 5)
            # if pair are in the same row take the letter to the right of each letter to construct cipher
            if row1 == row2:
                ciphertext += table[row1 * 5 + (col1 + 1) % 5] 
                ciphertext += table[row2 * 5 + (col2 + 1) % 5]
            elif col1 == col2:  # if pair are in the same column take the letter below of each letter to construct cipher
                ciphertext += table[((row1 + 1) % 5) * 5 + col1]
                ciphertext += table[((row2 + 1) % 5) * 5 + col2]
            else: 
                ciphertext += table[row1 * 5 + col2]
                ciphertext += table[row2 * 5 + col1]
    
        return ciphertext
    
    
    def decode(self,ciphertext, key):
        table = self.generate_table(key)
        print(f'Constructing the square key matrix\n\n{table}\n')
        plaintext = ""
    
        for char1, char2 in self.chunker(ciphertext, 2):    # Each pair identify which row and column the pair exists (identify pair position)
            row1, col1 = divmod(table.index(char1), 5)
            row2, col2 = divmod(table.index(char2), 5)
            # if pair are in the same row take the letter to the left of each letter to construct cipher
            if row1 == row2:
                plaintext += table[row1 * 5 + (col1 - 1) % 5]
                plaintext += table[row2 * 5 + (col2 - 1) % 5]
            elif col1 == col2:  # if pair are in the same column take the letter above of each letter to construct cipher
                plaintext += table[((row1 - 1) % 5) * 5 + col1]
                plaintext += table[((row2 - 1) % 5) * 5 + col2]
            else:
                plaintext += table[row1 * 5 + col2]
                plaintext += table[row2 * 5 + col1]
    
        return plaintext

class ColumnTransposition():
# Encryption
    def encryptMessage(self,msg):
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
    def decryptMessage(self,cipher):
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
            raise TypeError("Key not compatible with the algorithm",
                            "Change key to not include duplicate letters")
    
        null_count = msg.count('_')
    
        if null_count > 0:
            return msg[: -null_count]
    
        return msg

class AES():

    rounds_by_key_size = {16: 10, 24: 12, 32: 14}
    def __init__(self, master_key):
        """
        Initializes the object with a given key.
        """
        assert len(master_key) in AES.rounds_by_key_size
        self.n_rounds = AES.rounds_by_key_size[len(master_key)]
        self._key_matrices = self._expand_key(master_key)

    def _expand_key(self, master_key):
        """
        Expands and returns a list of key matrices for the given master_key.
        """
        # Initialize round keys with raw key material.
        key_columns = bytes2matrix(master_key)
        iteration_size = len(master_key) // 4

        i = 1
        while len(key_columns) < (self.n_rounds + 1) * 4:
            # Copy previous word.
            word = list(key_columns[-1])

            # Perform schedule_core once every "row".
            if len(key_columns) % iteration_size == 0:
                # Circular shift.
                word.append(word.pop(0))
                # Map to S-BOX.
                word = [s_box[b] for b in word]
                # XOR with first byte of R-CON, since the others bytes of R-CON are 0.
                word[0] ^= r_con[i]
                i += 1
            elif len(master_key) == 32 and len(key_columns) % iteration_size == 4:
                # Run word through S-box in the fourth iteration when using a
                # 256-bit key.
                word = [s_box[b] for b in word]

            # XOR with equivalent word from previous iteration.
            word = xor_bytes(word, key_columns[-iteration_size])
            key_columns.append(word)

        # Group key words in 4x4 byte matrices.
        return [key_columns[4*i : 4*(i+1)] for i in range(len(key_columns) // 4)]

    def encrypt_block(self, plaintext):
        """
        Encrypts a single block of 16 byte long plaintext.
        """
        assert len(plaintext) == 16

        plain_state = bytes2matrix(plaintext)

        add_round_key(plain_state, self._key_matrices[0])

        for i in range(1, self.n_rounds):
            sub_bytes(plain_state)
            shift_rows(plain_state)
            mix_columns(plain_state)
            add_round_key(plain_state, self._key_matrices[i])

        sub_bytes(plain_state)
        shift_rows(plain_state)
        add_round_key(plain_state, self._key_matrices[-1])

        return matrix2bytes(plain_state)

    def decrypt_block(self, ciphertext):
        """
        Decrypts a single block of 16 byte long ciphertext.
        """
        assert len(ciphertext) == 16

        cipher_state = bytes2matrix(ciphertext)

        add_round_key(cipher_state, self._key_matrices[-1])
        inv_shift_rows(cipher_state)
        inv_sub_bytes(cipher_state)

        for i in range(self.n_rounds - 1, 0, -1):
            add_round_key(cipher_state, self._key_matrices[i])
            inv_mix_columns(cipher_state)
            inv_shift_rows(cipher_state)
            inv_sub_bytes(cipher_state)

        add_round_key(cipher_state, self._key_matrices[0])

        return matrix2bytes(cipher_state)

    def encrypt_cbc(self, plaintext, iv):

        assert len(iv) == 16 # Assertion to ensure that the initial vector is 16

        plaintext = pad(plaintext) # pad the given plaintext PKCS#7

        blocks = []
        previous = iv
        for plaintext_block in split_blocks(plaintext):
            # CBC mode encrypt: encrypt(plaintext_block XOR previous)
            block = self.encrypt_block(xor_bytes(plaintext_block, previous))
            blocks.append(block)
            previous = block

        return b''.join(blocks)

    def decrypt_cbc(self, ciphertext, iv):

        assert len(iv) == 16

        blocks = []
        previous = iv
        for ciphertext_block in split_blocks(ciphertext):
            # CBC mode decrypt: previous XOR decrypt(ciphertext)
            blocks.append(xor_bytes(previous, self.decrypt_block(ciphertext_block)))
            previous = ciphertext_block

        return unpad(b''.join(blocks))

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Driver code for benchmarking algorithms
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Parsing and preparing input text file
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
print('This is a benchmark to time classical and modern cipher algorithms\n')
print('Input file: "plaintext.txt"\n')
tokenizer = Tokenizer('plaintext.txt','cleantext.txt')
print("Cleaning input file ...\n")
tokenizer.cleaner()
print("parsing cleaned input file ...\n\nThe message is\n")
tokens = tokenizer.readFile()
print(tokenizer.concatList(tokens))
tokensstring = tokenizer.concatList(tokens)
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
print('\nPerforming the Affine encryption algorithm ...\n')
affineres = []
affinestart = time.time()
for token in range(len(tokens)):
    affineres.append(Affine().encrypt(5,8,tokens[token],ALPHABET_SIZE))
print(tokenizer.concatList(affineres))
affineend = time.time()
print(f'\nTime taken to perform the affine encryption algrithm is {(affineend-affinestart)*1000} milliseconds')
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
print('\nPerforming the Affine decryption algorithm ...\n')
affinedecres = []
affinedecstart = time.time()
for token in range(len(affineres)):
    affinedecres.append(Affine().decrypt(5,8,affineres[token],ALPHABET_SIZE))
print(tokenizer.concatList(affinedecres))
affinedecend = time.time()
print(f'\nTime taken to perform the affine decryption algrithm is {(affinedecend-affinedecstart)*1000} milliseconds')
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
print("\nPerforming the PlayFair encryption algorithm ...\n")
playfairres = []
playfairstart = time.time()
playfairres=PlayFair().encode(tokensstring,key)
playfairend = time.time()
print(playfairres)
print(f'\nTime taken to perform the playfair encryption algrithm is {(playfairend-playfairstart)*1000} milliseconds')
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
print("\nPerforming the PlayFair decryption algorithm ...\n")
playdecfairres = []
playfairdecstart = time.time()
playdecfairres=PlayFair().decode(playfairres,key)
playfairdecend = time.time()
print(playdecfairres+"\n")
print(f'\nTime taken to perform the playfair decryption algrithm is {(playfairdecend-playfairdecstart)*1000} milliseconds')
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
print("\nPerforming the Column Transposition encryption algorithm ...\n")
columnres = []
columnstart = time.time()
columnres = ColumnTransposition().encryptMessage(tokensstring)
columnend = time.time()
print(columnres)
print(f'\nTime taken to perform the Column Transposition encryption algrithm is {(columnend-columnstart)*1000} milliseconds')
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
print("\nPerforming the Column Transposition decryption algorithm ...\n")
columndecstart = time.time()
columndecres = ColumnTransposition().decryptMessage(columnres)
columndecend = time.time()
print(columndecres)
print(f'\nTime taken to perform the Column Transposition decryption algrithm is {(columndecend-columndecstart)*1000} milliseconds')
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
keyAes = os.urandom(16)
iv = os.urandom(16)
print("\nPerforming the AES with CBC encryption algorithm ...\n")
aesstart = time.time()
encrypted = AES(keyAes).encrypt_cbc(bytes(tokensstring,'utf-8'), iv)
aesend = time.time()
print(f"\n{encrypted}")
print(f'\nTime taken to perform the AES encryption algrithm is {(aesend-aesstart)*1000} milliseconds')
print('\n')
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
print("\nPerforming the AES with CBC decryption algorithm ...\n")
aesdecst = time.time()
decrypted = AES(keyAes).decrypt_cbc(encrypted,iv)
aesdecend = time.time()
print(f"\n{decrypted}")
print(f'\nTime taken to perform the AES decryption algrithm is {(aesdecend-aesdecst)*1000} milliseconds')
print('\n')
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------