class Note():
    '''
    Contains:
        - tone type
        - note start time
        - note length
        - note frequency (can be vector, indicates frequency changing over time, in Hz)
        - fs


    '''
    def __init__(self):
        pass
    def gen_sig(self):
        pass


class Tone():
    '''
    Contains amplitude generation method.
    Elements:
    Base Path
    amplitude response


    Methods:
        -Multiply by scalar
        -modulate by another tone (multiply)
        -add tones
        -add scalar
    '''