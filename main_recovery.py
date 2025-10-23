from langchain_text_splitters import RecursiveCharacterTextSplitter,MarkdownTextSplitter,Language,PythonCodeTextSplitter,NLTKTextSplitter
from langchain_core.documents import Document
import re
import os
# Example long text to split
text = """
This is a sample long document. It contains multiple paragraphs.

Each paragraph has its own content. The RecursiveCharacterTextSplitter will split this text based on the specified separators.

It tries to split on double newlines first, then single newlines, then spaces, and finally characters if needed.

This ensures that the text is split in a way that preserves as much structure as possible.
"""

arabicText = """
هذا مستند طويل نموذجي. يحتوي على فقرات متعددة.
كل فقرة لها محتواها الخاص. سيقوم RecursiveCharacterTextSplitter بتقسيم هذا النص بناءً على الفواصل المحددة.
يحاول التقسيم على الفواصل المزدوجة أولاً، ثم الفواصل الفردية، ثم المسافات، وأخيراً الأحرف إذا لزم الأمر.
يضمن ذلك تقسيم النص بطريقة تحافظ على أكبر قدر ممكن من الهيكل.
"""

markdownText = """
# Initialize the RecursiveCharacterTextSplitter

## Overview
The RecursiveCharacterTextSplitter is a powerful tool for breaking down large documents into smaller, manageable chunks while preserving the structure and context of the original text.

## Key Features
- **Hierarchical Splitting**: Uses a hierarchy of separators to maintain document structure
- **Customizable Chunk Size**: Control the maximum size of each text chunk
- **Overlap Control**: Specify overlap between chunks to maintain context
- **Multiple Separators**: Try different separators in order of preference

## Configuration Parameters
### chunk_size
The maximum number of characters in each chunk. Default is usually around 1000.

### chunk_overlap
The number of characters that overlap between consecutive chunks to maintain context.

### separators
A list of separators to try in order:
1. Double newlines (`\\n\\n`) - for paragraph breaks
2. Single newlines (`\\n`) - for line breaks
3. Spaces (` `) - for word breaks
4. Empty string (`""`) - for character-level splitting

## Best Practices
- Choose chunk_size based on your model's context window
- Use appropriate overlap (10-20% of chunk_size) to maintain context
- Test with your specific document types to optimize parameters
- Consider the downstream task when setting parameters

## Example Usage
```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=450,
    chunk_overlap=20,
    separators=["\\n\\n", "\\n", " ", ""]
)
```
"""


def word_count_length(text):
    return len(text.split())
def sentence_count_length(text):
    """
    Improved sentence counting function that handles edge cases
    """
    if not text or not text.strip():
        return 0
    
    # Split by periods and filter out empty strings
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    return len(sentences) # 3 sentences per chunk # 1 sentence overlap
# Alternative more robust sentence counting function
def robust_sentence_count_length(text):
    """
    More robust sentence counting using multiple sentence delimiters
    """
    if not text or not text.strip():
        return 0
    
    import re
    # Split by sentence-ending punctuation (., !, ?)
    sentences = re.split(r'[.!?]+', text)
    # Filter out empty strings and whitespace-only strings
    sentences = [s.strip() for s in sentences if s.strip()]
    return len(sentences)

def advanced_sentence_count(text):
    """Advanced sentence counting with abbreviation handling"""
    if not text or not text.strip():
        return 0
    
    # Common abbreviations that shouldn't split sentences
    abbreviations = ['Dr.', 'Mr.', 'Mrs.', 'Ms.', 'Prof.', 'etc.', 'i.e.', 'e.g.', 'vs.']
    
    # Replace abbreviations temporarily
    temp_text = text
    for i, abbr in enumerate(abbreviations):
        temp_text = temp_text.replace(abbr, f'ABBREV{i}')
    
    # Split by sentence endings
    sentences = re.split(r'[.!?]+', temp_text)
    
    # Restore abbreviations
    for i, abbr in enumerate(abbreviations):
        temp_text = temp_text.replace(f'ABBREV{i}', abbr)
    
    return len([s.strip() for s in sentences if s.strip()])

advanced_sentence_example_text='''
Mr. Smith went to Washington. He met with Dr. Brown to discuss the project. The meeting was productive!

'''

#. 1) Create the text splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,  # Maximum size of each chunk
    chunk_overlap=40,  # Overlap between chunks
    # separators=["\n\n", "\n", " ", ""],  # Hierarchy of separators to try
    # strip_whitespace=True,
    length_function=advanced_sentence_count,
    # separators=[r"\n\n", r"\n", r"\. ", r" "],
    # is_separator_regex=True
    # separators=[". ", "! ", "? ", "\n\n", "\n", " "]
)

# 1. Basic NLTKTextSplitter usage
nltk_splitter = NLTKTextSplitter(
    chunk_size=200,  # Maximum characters per chunk
    chunk_overlap=40,  # Overlap between chunks
    # separator="\n",  # You can specify a separator if needed
    # is_separator_regex=False
    strip_whitespace=True,
    # length_function=word_count_length,
)




# Convert chunks into Document objects
# print(splitter.create_documents([advanced_sentence_example_text]))
# print(nltk_splitter.create_documents([advanced_sentence_example_text]))



# 2) Create markdown documentation
mspllitter = MarkdownTextSplitter(
    chunk_size=500,  # Maximum size of each chunk
    chunk_overlap=50,  # Overlap between chunks
    # saparators=["\n\n", "\n", " ", ""]
)
# print(mspllitter.create_documents([markdownText]))

# Python code splitter
samplePython = '''
def example_function(param1, param2):
    """
    This is an example function that demonstrates how to split Python code into chunks.
    It takes two parameters and returns their sum.
    """
    result = param1 + param2
    return result
    '''
pySplitter = PythonCodeTextSplitter(
    chunk_size=400,
    chunk_overlap=40)

# print(pySplitter.create_documents([samplePython]))


# Different Language Splitter

jsLanguageText = '''
function greet(name) {
    // This function greets the user by name
    console.log("Hello, " + name + "!");
}
greet("World");
'''
goLanguageText = '''
package main
import "fmt"
// This program prints Hello, World! to the console
func main() {
    fmt.Println("Hello, World!")
}
'''
htmlLanguageText = '''<!DOCTYPE html>
<html>
<head>
    <title>Sample HTML</title>
</head>
<body>
    <h1>Welcome to the Sample HTML Page</h1>
    <p>This is a paragraph in the HTML document.</p>
</body>
</html>
'''

typeScriptLanguageText = '''
function add(a: number, b: number): number {
    // This function adds two numbers and returns the result
    return a + b;
}
'''

langSplitter = RecursiveCharacterTextSplitter.from_language(
    chunk_size=100,
    chunk_overlap=60,
    language=Language.MARKDOWN,
    length_function=advanced_sentence_count,
    # separators=[". ", "! ", "? ", "\n\n", "\n", " "],
    strip_whitespace=True,
    # is_separator_regex=True
    )

# print(langSplitter.create_documents([markdownText]))

# Sliding Window Splitter with LangChain Document objects
from utils import SlidingWindowSplitter,AgenticChunker,AgenticChunker2
text = '''
This is a sample text to demonstrate the SlidingWindowSplitter. It will split this text into smaller chunks based on the specified window size and step size.'''

    # Character-level
char_splitter = SlidingWindowSplitter(
    window_size=200,    # Size of each chunk
    step_size=100,      # How far to move the window (50% overlap)
    granularity="sentence", # Unit of measurement
)
# print(char_splitter.create_documents([text]))


key = os.getenv("OPENAI_API_KEY")
ac = AgenticChunker2(openai_api_key=key, model="gpt-3.5-turbo", temperature=0.0, print_logging=True)

example_props = [
        "The month is October.",
        "The year is 2023.",
        "One of the most important things that I didn't understand about the world as a child was the degree to which the returns for performance are superlinear.",
        "Teachers and coaches implicitly told us that the returns were linear.",
        "I heard a thousand times that 'You get out what you put in.'",
    ]

ac.add_propositions(example_props)
ac.pretty_print_chunks()

docs = ac.get_chunks_as_documents()
print(f"\nProduced {len(docs)} Document objects.")
for d in docs:
    print(d.metadata)

    

