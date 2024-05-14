# LDA Analysis of Negative Comments

This Python project applies Latent Dirichlet Allocation (LDA) to analyze negative comments and uncover prevailing themes or topics. The script processes textual data to perform natural language processing, executes preprocessing, and utilizes the LDA model from the Gensim library to identify and rank topics based on their prominence in the dataset.

## Requirements

This project is developed using Python and requires the following libraries:

- pandas
- gensim
- nltk

Ensure Python 3.6 or higher is installed on your system.

## Installation

Follow these steps to set up the project environment and run the script:

### Clone the Repository

First, clone this repository to your local machine using:

git clone https://github.com/yourusername/lda-negative-comments.git
cd lda-negative-comments

### Install Required Libraries

Install the necessary Python libraries using pip:

pip install pandas gensim nltk matplotlib

### Download NLTK Resources

After installing the nltk library, you will need to download specific resources used by the script:

import nltk
nltk.download('stopwords')
nltk.download('punkt')

## Usage

### Data Preparation

Ensure your dataset is in a CSV format with negative comments in the first column and no header. Adjust the script if your data format differs.

### Running the Script

Execute the script by running:

python lda_analysis.py

The script will load the data, perform LDA, and print the ranked topics based on their distribution.

## Output

The script outputs the topics sorted from the most to the least mentioned, including the top terms in each topic with their respective weights. This output helps in understanding the major themes discussed in the negative comments.

## Contributing

Contributions to this project are welcome. Please fork the repository and submit a pull request with your enhancements.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
