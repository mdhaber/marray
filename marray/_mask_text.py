import re
import textwrap

# Regular expression generated by ChatGPT
# Capture everything up to and including the first "[", "]", or ","
pattern = r"[^,\[\]\),]*(?:[\[\]\),])"

pattern = re.compile(pattern, re.VERBOSE)

def _mask_repr(text, mask):
    # special cases
    if not mask.size:  # nothing to mask
        return text
    elif mask.ndim == 0:  # no square brackets or commas to parse
        if mask:
            text = re.sub(r"(?<=\().*?(?=[,\)])", "_", text)
        return text

    out = []  # store the output string
    index = [0] * mask.ndim  # index of the current element
    dim = -1  # keep track of which dimension of index we're incrementing
    while(text):
        match = pattern.search(text)

        match_text = match.group(0)
        text = text[match.end():]

        if match_text[-1] == "[":
            dim += 1

        if re.search(r"\d|nan|inf|False|True", match_text):  # number
            if mask[tuple(index)]:
                # this could use improvement. Might be nice to center "__"?
                first_char = match_text[0] if match_text[0].isspace() else " "
                match_text = first_char + " "*(len(match_text) - 3) + "_" + match_text[-1]
        elif re.search(r"\.\.\.", match_text):  # ellipses
            # assumes there are equal number of elements
            # (e.g. values, rows) before and after ellipses
            index[dim] *= -1
            index[dim] -= 1

        if match_text[-1] == "]":
            index[dim] = 0
            dim -= 1
        elif match_text[-1] == ",":
            index[dim] += 1

        out.append(match_text)

        # Exit after reaching the last bracket so array metadata
        # (e.g. "shape=(2, 3, 1000)") isn't processed
        if dim == -1:
            break

    out.append(text)
    return "".join(out)

def _mask_str(text, mask):
    if mask.ndim == 0:
        text = re.search(r"(?<=\().*?(?=[\),])", text).group(0)
        return "_" if mask else text
    text = _mask_repr(text, mask)
    left = text.find("[")
    right = len(text) - text[::-1].find("]")
    text = " "*left + text[left:right]
    text = text.replace(",", "")
    return textwrap.dedent(text)
