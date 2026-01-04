#1
from Bio import Entrez
import time
from typing import List, Dict
class PubMedFetcher:
    def __init__(self, email: str):
        Entrez.email = email
    def search_papers(self, query: str, max_results: int = 100) -> List[str]:
        """搜索文献返回PMID列表"""
        handle = Entrez.esearch(
            db="pubmed",
            term=query,
            retmax=max_results,
            sort="relevance"
        )
        record = Entrez.read(handle)
        handle.close()
        return record["IdList"]
    def fetch_abstracts(self, pmid_list: List[str]) -> List[Dict]:
        """批量获取摘要"""
        papers = []
        # 分批处理（避免API限流）
        batch_size = 10
        for i in range(0, len(pmid_list), batch_size):
            batch = pmid_list[i:i+batch_size]
            handle = Entrez.efetch(
                db="pubmed",
                id=",".join(batch),
                rettype="abstract",
                retmode="xml"
            )
            records = Entrez.read(handle)
            handle.close()
            for article in records['PubmedArticle']:
                try:
                    medline = article['MedlineCitation']
                    article_data = medline['Article']
                    # 提取关键信息
                    paper = {
                        'pmid': str(medline['PMID']),
                        'title': article_data['ArticleTitle'],
                        'abstract': article_data.get('Abstract', {}).get('AbstractText', [''])[0],
                        'journal': article_data['Journal']['Title'],
                        'year': article_data['Journal']['JournalIssue'].get('PubDate', {}).get('Year', 'N/A'),
                        'authors': self._extract_authors(article_data.get('AuthorList', []))
                    }
                    if paper['abstract']:  # 只保留有摘要的
                        papers.append(paper)
                        print(f"✅ 获取: {paper['title'][:50]}...")
                except Exception as e:
                    print(f"❌ 解析失败: {e}")
            time.sleep(0.5)  # 避免限流
        return papers
    def _extract_authors(self, author_list) -> str:
        """提取作者名"""
        authors = []
        for author in author_list[:3]:  # 只取前3位
            if 'LastName' in author and 'Initials' in author:
                authors.append(f"{author['LastName']} {author['Initials']}")
        return ", ".join(authors) + (" et al." if len(author_list) > 3 else "")

#2
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document 
from langchain_community.vectorstores import Chroma
from typing import List, Dict

class MedicalRAGBuilder:
    def __init__(self, 
                 embedding_model: str = ""):
        print(f" 加载Embedding模型: {embedding_model}")
        #embedding模型
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model, 
            model_kwargs={'device': 'cpu'}  # 如果有GPU改为'cuda'
        )
        #文本分割器
        self.text_splitter = RecursiveCharacterTextSplitter(
            #递归字符分割：按不同的字符递归地分割(按照separators中的优先级:"\n\n", "\n", "." , ……)
            chunk_size=500,  # 每个chunk大小
            chunk_overlap=50,  # 相邻chunk之间的重叠字符数，防止上下文丢失
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""] #分隔符字符串数组
        )
        #向量数据库
        self.vectorstore = None

#!rm -rf './chroma_db'  # 删除旧的数据库文件（如果文件夹中有文件的话），windows电脑请手动删除
    def build_vectorstore(self, papers: List[Dict], 
                          #定义持久化路径persist_directory
                          persist_directory: str = ""):
        """构建向量数据库"""
        documents = []
        for paper in papers:
            # 构造文档内容
            content = f"Title: {paper['title']}\n\n"
            content += f"PMID: {paper['pmid']}\n\n"
            content += f"Abstract: {paper['abstract']}\n\n"
            content += f"Journal: {paper['journal']} ({paper['year']})\n"
            content += f"Authors: {paper['authors']}"
            # 创建Document对象: from langchain_core.documents import Document
            doc = Document(
                #内容page_content
                page_content=content,
                #描述性数据metadata
                metadata={
                    'pmid': paper['pmid'],
                    'title': paper['title'],
                    'year': paper['year'],
                    'source': f"https://pubmed.ncbi.nlm.nih.gov/{paper['pmid']}/"
                }
            )
            documents.append(doc)
        print(f" 准备对 {len(documents)} 篇文献进行向量化...")

        # 分割文本
        split_docs = self.text_splitter.split_documents(documents)
        print(f"✂️ 分割为 {len(split_docs)} 个文本块")
        # 向量数据库
        self.vectorstore = Chroma.from_documents(
            documents=split_docs,
            embedding=self.embeddings,
            persist_directory=persist_directory
        )
        self.vectorstore.persist()
        print(f"向量库中存储的数量：{self.vectorstore._collection.count()}")
        print(f"✅ 向量数据库已保存到: {persist_directory}")
    def load_vectorstore(self, 
                         persist_directory: str = ""):
        """加载已有的向量数据库"""
        self.vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embeddings
        )
        print(f"✅ 向量数据库已加载")
#向量检索
#当你需要数据库返回严谨的 按余弦相似度排序的结果 时可以使用similarity_search函数。
    def similarity_search(self, query: str, k: int = 5):
        """相似度检索"""
        if not self.vectorstore:
            raise ValueError("请先构建或加载向量数据库")
        #返回按余弦相似度排序的前k个文献片段
        results = self.vectorstore.similarity_search_with_score(query, k=k)
        print(f"\n 检索问题: {query}")
        print(f" 找到 {len(results)} 个相关文献片段:\n")

        for i, (doc, score) in enumerate(results):
            print(f"检索到的第{i}个文献片段的: \n")
            print(f"   相似度: {score:.4f}")
            print(f"   标题: {doc.metadata['title']}")
            print(f"   内容: {doc.page_content[:200]}...")
            print(f"   来源: {doc.metadata['source']}\n")
        return results

# 防止AI生成的答案中引用的PMID不存在
def verify_citations(answer: str, source_docs: List) -> str:
    """验证并修正引用"""
    valid_pmids = [doc.metadata['pmid'] for doc in source_docs]
    # 提取答案中的PMID
    import re
    mentioned_pmids = re.findall(r'PMID:\s*(\d+)', answer)
    # 过滤无效引用
    for pmid in mentioned_pmids:
        if pmid not in valid_pmids:
            answer = answer.replace(f"PMID: {pmid}", "[引用验证失败]")
    return answer

#3.
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())  

class MedicalQASystem:
    def __init__(self, vectorstore):
        """
        初始化问答系统
        Args:model_name: 可选
        """
        self.vectorstore = vectorstore

        #⭐构建检索问答链
        #question = ""
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
        #docs = retriever.invoke(question)
        #print(f"检索到的内容数：{len(docs)}")
        #for i, doc in enumerate(docs):
           #print(f"检索到的第{i}个内容: \n {doc.page_content}", end="\n-------------\n")
        #⭐配置LLM
        """
        self.llm = OpenAI(
            model_name=model_name,
            temperature=0.3,  # 降低随机性，提高准确度
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )"""
        model_name = os.environ['MODEL_NAME']
        api_key = os.environ['API_KEY'] 
        base_url = os.environ['BASE_URL'] 
        # print(f"MODEL_NAME={model_name}, API={api_key}, BASE_URL={base_url}")
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0.3,                      # temperature↓，随机性减少
            openai_api_key=api_key,      
            openai_api_base=base_url
        )
        # ⭐PromptTemplates👇
        self.template = """你是一位专业的医学文献分析助手。请基于以下文献内容回答问题。
要求：
1. 答案必须基于提供的文献内容
2. 引用具体的PMID和文献标题
3. 如果文献中没有相关信息，明确说明
4. 使用专业但易懂的语言
5. 注意：所有缩写按医学术语解释（如MI=心肌梗死）
文献内容：
{context}
问题：{question}
请给出详细的答案："""
        self.prompt = PromptTemplate(template=self.template)
        #⭐构建QA链
        self.qa_chain = (
    RunnableParallel(
        {
            "docs": retriever,
            "question": RunnablePassthrough()
        }
    )
    | RunnableParallel(
        {
            "answer": (
                RunnableLambda(
                    lambda x: {
                        "context": "\n\n".join(d.page_content for d in x["docs"]),
                        "question": x["question"],
                    }
                )
                | self.prompt
                | self.llm
                | StrOutputParser()
            ),
            "source_docs": lambda x: x["docs"],
        }
    )
)

    def ask(self, question: str) :# -> Dict
        """
        提问并获取答案（检索问答链 效果测试）
        Returns:
            {
                'answer': str,
                'sources': List[Dict]
            }
        """
        print(f"\n 问题: {question}\n")
        print(" AI正在思考...\n")
        self.result = self.qa_chain.invoke(question)
        
        self.answer = self.result["answer"]
        self.source_docs = self.result["source_docs"]
        #优化：验证引用的PMID是否真实存在
        self.valid_answer = verify_citations(self.answer, self.source_docs)
        
        # 格式化输出
        print(" 答案:")
        print("-" * 80)
        print(self.valid_answer)
        print("-" * 80)
        print("\n 参考文献:")
        for i, doc in enumerate(self.source_docs, 1):
            title = doc.metadata.get("title", "Unknown title")
            pmid = doc.metadata.get("pmid", "Unknown PMID")
            source = doc.metadata.get("source", "Unknown source")
            print(f"\n[{i}] {title}")
            print(f"    PMID: {pmid}")
            print(f"    来源: {source}")

        return {
            'answer': self.valid_answer,
            'sources': [doc.metadata for doc in self.source_docs]
        }
    
#完整流程
if __name__ == "__main__":
    # Step 1: 获取文献
    fetcher = PubMedFetcher(email="eyl998600@gmail.com")
    pmids = fetcher.search_papers("Alzheimer's disease treatment 2023", max_results=50)
    papers = fetcher.fetch_abstracts(pmids)
    # Step 2: 构建向量数据库
    rag = MedicalRAGBuilder()
    #rag.build_vectorstore(papers, persist_directory="./chroma_db")
    if os.path.exists("./chroma_db"):
       rag.load_vectorstore()
    else: #否则每次运行都会重新建库
       rag.build_vectorstore(papers)
    # Step 3: 创建问答系统
    qa_system = MedicalQASystem(rag.vectorstore)
    # Step 4: 提问
    questions = [
        "What are the most promising treatments for Alzheimer's disease in 2023?",
        "What is the mechanism of action of aducanumab?",
        "Are there any clinical trials showing positive results?"
    ]
    for q in questions:
        qa_system.ask(q)
        print("\n" + "="*100 + "\n")
    

