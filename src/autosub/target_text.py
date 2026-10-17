"""Target-only text comparison with explicitly unscored surrounding context."""
from .language_validation import normalized_text


def target_distance(reference, hypothesis):
    a, b = normalized_text(reference), normalized_text(hypothesis)
    if not a or len(a)*len(b) > 4_000_000:
        raise ValueError('Empty or too large target')
    # Cost, negative exact-match count. Prefer more exact matches on equal cost.
    grid = [[(0, 0)] * (len(b)+1)]
    for i, char in enumerate(a, 1):
        row = [(i, 0)]
        for j, other in enumerate(b, 1):
            cost, negative_matches = grid[i-1][j-1]
            diagonal = (cost + (char != other), negative_matches - (char == other))
            deletion = (grid[i-1][j][0]+1, grid[i-1][j][1])
            insertion = (row[-1][0]+1, row[-1][1])
            row.append(min(diagonal, deletion, insertion))
        grid.append(row)
    end = min(range(len(b)+1), key=lambda j: grid[-1][j])
    i, j = len(a), end
    while i:
        if j and grid[i][j] == (grid[i-1][j-1][0] + (a[i-1] != b[j-1]), grid[i-1][j-1][1] - (a[i-1] == b[j-1])):
            i, j = i-1, j-1
        elif grid[i][j] == (grid[i-1][j][0]+1, grid[i-1][j][1]):
            i -= 1
        else:
            j -= 1
    return {'character_edits': grid[-1][end][0], 'reference_characters': len(a),
        'target_only_cer': grid[-1][end][0]/len(a), 'normalized_hypothesis_span': [j, end],
        'matched_hypothesis_text': b[j:end], 'unscored_context_characters': len(b)-(end-j),
        'scope': 'free leading/trailing context; cannot measure whole-file hallucinations or completion'}
