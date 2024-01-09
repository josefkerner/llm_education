import pandas as pd
from langchain.schema.document import Document
from utils.connectors.vectara_doc_loader import DocumentLoader
from tqdm import tqdm
class PrepareData:
    def __init__(self):
        self.data_path = 'data/marketing.csv'
        self.document_loader = DocumentLoader()

    def combine_description(self, row):
        description = row['About Product']
        if description == 'nan':
            description = ''
        description = str(description).replace('Make sure this fits your model number.', '')

        full_desc = f"""
        Product name: {row['Product Name']}
        Brand: {row['Brand Name']}
        Category: {row['Category']}
        Price: {row['Selling Price']}
        Description: {description}
        Color: {row['Color']}
        Specifications: {row['Product Specification']}
        
        """
        return full_desc

    def prepare(self):
        df = pd.read_csv(self.data_path)
        print(df.columns)
        #get main category from category column
        df['Major Category'] = df['Category'].str.split('|').str[0]
        #get value counts
        print(df['Major Category'].value_counts())
        #remove rows where Category is nan
        df = df[df['Category'].notna()]
        df = df[df['Category'].str.startswith('Sports')]
        #get full description
        df['full_description'] = df.apply(self.combine_description, axis=1)

        print(df['Category'].value_counts())
        #
        #count rows
        print(df.shape)

        #save to excel file
        df.to_excel('data/marketing_processed.xlsx')
        #iter rows
        docs = []
        for index, row in df.iterrows():
            #create document from full description
            doc = Document(
                page_content=row['full_description'],
                metadata={
                    'source': row['Product Name'],
                    'category': row['Category']
                }
            )
            docs.append(doc)
        return docs

    def add_docs(self, docs):
        document_loader = DocumentLoader()
        print('Adding docs to vector store')
        for doc in tqdm(docs):
            document_loader.add_doc_to_vector_store(doc)

if __name__ == "__main__":
    prepare_data = PrepareData()
    docs = prepare_data.prepare()
    #prepare_data.add_docs(docs)