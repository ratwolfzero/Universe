import math

def generate_sequence(length):
    """
    Generates a sequence defined by the recursive formula:
    a_0 = 19
    a_1 = 4 * a_0
    a_{n+1} = 4 * a_n + 4 * 2^(2 * (n-1)) for n >= 1
    
    Args:
        length (int): The number of terms to generate in the sequence.
        
    Returns:
        list: A list containing the terms of the sequence.
    """
    if length <= 0:
        return []
																					  
    sequence = [0] * length
    sequence[0] = 19
    
    if length > 1:
        sequence[1] = 4 * sequence[0]
        
    for n in range(1, length - 1):
        # a_{n+1} = 4 * a_n + 4 * 2^(2 * (n-1))
        sequence[n+1] = 4 * sequence[n] + 4 * math.pow(2, 2 * (n-1))
        
    return sequence


def generate_sequence_closed_form(length):
    """
    Generates a sequence using the closed-form formula:
    a_n = (18.75 + n/4) * 4^n for n >= 1
    
    Args:
        length (int): The number of terms to generate in the sequence.
        
    Returns:
        list: A list containing the terms of the sequence.
    """
    if length <= 0:
        return []
    
    sequence = [0] * length
    sequence[0] = 19
    
    for n in range(1, length):
        # a_n = (18.75 + n/4) * 4^n
        sequence[n] = (18.75 + n/4) * math.pow(4, n)
        
    return sequence

# Example usage:
sequence_length = 5
recursive_sequence = generate_sequence(sequence_length)
closed_form_sequence = generate_sequence_closed_form(sequence_length)

print(f"Sequence using recursive formula (length {sequence_length}):")
print(recursive_sequence)
print("\n" + "="*50 + "\n")
print(f"Sequence using closed-form formula (length {sequence_length}):")
print(closed_form_sequence)
