import os
import io
import token
import tokenize

def remove_comments(source_code):
    io_obj = io.StringIO(source_code)
    out = ""
    prev_toktype = token.INDENT
    last_lineno = -1
    last_col = 0
    
    for tok in tokenize.generate_tokens(io_obj.readline):
        token_type = tok[0]
        token_string = tok[1]
        start_line, start_col = tok[2]
        end_line, end_col = tok[3]
        
        if start_line > last_lineno:
            last_col = 0
            
        if start_col > last_col:
            out += (" " * (start_col - last_col))
            
        if token_type == tokenize.COMMENT:
            pass # ignore comments
        elif token_type == tokenize.STRING:
            # check if it's a docstring
            if prev_toktype != token.INDENT and prev_toktype != token.NEWLINE and start_col > 0:
                out += token_string
            else:
                # likely a docstring, ignore it if it contains Chinese
                import re
                if re.search(r'[\u4e00-\u9fff]', token_string) or token_string.startswith('"""') or token_string.startswith("'''"):
                    pass # ignore multiline strings that are docstrings
                else:
                    out += token_string
        else:
            out += token_string
            
        prev_toktype = token_type
        last_lineno = end_line
        last_col = end_col
        
    return out

def process_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except:
        return
        
    if filepath.endswith('.py'):
        new_content = remove_comments(content)
        # also remove empty lines
        lines = [line for line in new_content.split('\n') if line.strip() != '']
        new_content = '\n'.join(lines) + '\n'
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)
    elif filepath.endswith('.sh') or filepath.endswith('.md'):
        new_lines = []
        import re
        for line in content.split('\n'):
            # remove Chinese chars in .sh or .md
            if re.search(r'[\u4e00-\u9fff]', line):
                line = re.sub(r'[\u4e00-\u9fff]', '', line)
            new_lines.append(line)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(new_lines))

if __name__ == '__main__':
    for root, dirs, files in os.walk(r"d:\code\ours\IEEE-TDSC-code\Clara"):
        for file in files:
            if file.endswith(('.py', '.sh', '.md')):
                process_file(os.path.join(root, file))
