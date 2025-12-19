def fen_style_to_number(s):
    d = {
        # White pieces
        'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,
        # Black pieces
        'p': 7, 'n': 8, 'b': 9, 'r': 10, 'q': 11, 'k': 12,
    }
    return d.get(s, 0)  # return 0 for empty squares or unknown

def token_style_to_number(s):
    d = {
        'empty': 0,
        'pw': 1, 'nw': 2, 'bw': 3, 'rw': 4, 'qw': 5, 'kw': 6,
        'pb': 7, 'nb': 8, 'bb': 9, 'rb': 10, 'qb': 11, 'kb': 12,
    }
    return d.get(s, 0)  # return 0 for empty squares or unknown

def number_to_fen_style(n):
    d = {
        # White pieces
        1: 'P', 2: 'N', 3: 'B', 4: 'R', 5: 'Q', 6: 'K',
        # Black pieces
        7: 'p', 8: 'n', 9: 'b', 10: 'r', 11: 'q', 12: 'k',
    }
    return d.get(n, '.')  # return '.' for empty squares or unknown

def number_to_token_style(n):
    d = {
        0: 'empty',
        1: 'pw', 2: 'nw', 3: 'bw', 4: 'rw', 5: 'qw', 6: 'kw',
        7: 'pb', 8: 'nb', 9: 'bb', 10: 'rb', 11: 'qb', 12: 'kb',
    }
    return d.get(n, 'empty')  # return 'empty' for empty squares or unknown
