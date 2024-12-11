import os
import re
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import matplotlib.pyplot as plt

# Bước 1: Đọc dữ liệu từ tệp
folder_path_train = '/Users/K.Thien_LAP/NNTN/NLP_SUMMARY/TEST/train'
output_file_path = 'C:/Users/K.Thien_LAP/NNTN/NLP_SUMMARY/TEST/output.txt'

with open(output_file_path, 'w', encoding='utf-8') as output_file:
    with open(r'C:/Users/K.Thien_LAP/NNTN/NLP_SUMMARY/TEST/train/d064j', 'r', encoding='utf-8') as file:
        content = file.read()
        output_file.write(content)  # Ghi nội dung vào tệp

print(f"Dữ liệu đã được lưu vào {output_file_path}")

# Bước 2: Tiền xử lý văn bản
fp = open(output_file_path, 'r')
out_put_data = fp.read()
fp.close()

# Loại bỏ thẻ XML và lấy văn bản
soup = BeautifulSoup(out_put_data, 'html.parser')
text = soup.get_text()

# Tiền xử lý: chuyển thành chữ thường, loại bỏ ký tự đặc biệt và từ dừng
stop_words = set(stopwords.words('english'))

def preprocess_text(sentence):
    sentence = sentence.lower()  # Chuyển thành chữ thường
    sentence = re.sub(r'[^\w\s]', '', sentence)  # Loại bỏ ký tự đặc biệt
    tokens = word_tokenize(sentence)  # Tokenize câu
    filtered_tokens = [word for word in tokens if word not in stop_words]  # Loại bỏ từ dừng
    return " ".join(filtered_tokens)

# Tách câu và tiền xử lý
sentences = sent_tokenize(text)
processed_sentences = [preprocess_text(sentence) for sentence in sentences]

# Tính toán TF-IDF và độ tương đồng cosine
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(processed_sentences)

# Tính ma trận độ tương đồng cosine
cosine_sim_matrix = cosine_similarity(tfidf_matrix)
print("Ma trận độ tương đồng cosine:")
print(cosine_sim_matrix)

np.savetxt('cosine_similarity_matrix.csv', cosine_sim_matrix, delimiter=',')

# Xây dựng đồ thị từ ma trận độ tương đồng
graph = nx.from_numpy_array(cosine_sim_matrix)

# plt.figure(figsize=(10, 10))
# nx.draw(graph, with_labels=True, font_size=8, node_size=500, node_color="skyblue")
# plt.title("Đồ thị ma trận độ tương đồng")
# plt.show()

# Tính PageRank từ đồ thị
pagerank_scores = nx.pagerank(graph)

# Sắp xếp các câu theo điểm PageRank
ranked_sentences = sorted(((pagerank_scores[i], s) for i, s in enumerate(processed_sentences)), reverse=True)

# Tính số câu tóm tắt (30% số câu gốc)
total_sentences = len(ranked_sentences)
summary_ratio = 0.3
num_sentences_summary = max(1, int(total_sentences * summary_ratio))  # Đảm bảo có ít nhất 1 câu

# Lấy các câu tóm tắt
summary_sentences = [ranked_sentences[i][1] for i in range(num_sentences_summary)]

# Lưu kết quả tóm tắt vào tệp
Sum_auto_path = 'C:/Users/K.Thien_LAP/NNTN/NLP_SUMMARY/TEST/SUM_AUTO.txt'
with open(Sum_auto_path, 'w', encoding='utf-8') as file:
    for sentence in summary_sentences:
        file.write(sentence + '\n')

print(f"Dữ liệu tóm tắt đã được lưu vào {Sum_auto_path}")

# Bước 3: So sánh tóm tắt tự động với tóm tắt tham chiếu
output_file_path_SUM = 'C:/Users/K.Thien_LAP/NNTN/NLP_SUMMARY/TEST/outputSUM.txt'

with open(output_file_path_SUM, 'w', encoding='utf-8') as output_file:
    with open(r'C:/Users/K.Thien_LAP/NNTN/NLP_SUMMARY/TEST/train/d064j', 'r', encoding='utf-8') as file:
        content = file.read()
        output_file.write(content)  # Ghi nội dung vào tệp tham chiếu

# Đọc tệp tham chiếu
fp = open(output_file_path_SUM, 'r')
out_put_data_SUM = fp.read()
fp.close()

# Loại bỏ thẻ XML và lấy văn bản tham chiếu
soup = BeautifulSoup(out_put_data_SUM, 'html.parser')
text_SUM = soup.get_text()

# Tiền xử lý tham chiếu
processed_sentences_SUM = [preprocess_text(sentence) for sentence in sent_tokenize(text_SUM)]

# So sánh tóm tắt tự động và tóm tắt tham chiếu
vectorizer_SUM = TfidfVectorizer()
tfidf_matrix_SUM = vectorizer_SUM.fit_transform(processed_sentences_SUM)

# Tính độ tương đồng giữa các tóm tắt
similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix_SUM)

# Tính toán phần trăm câu giống nhau
set1 = set(summary_sentences)
set2 = set(processed_sentences_SUM)

common_sentences = set1 & set2
common_count = len(common_sentences)

print("Các câu giống nhau:")
print("\n".join(common_sentences))

percent_dataset1 = (common_count / len(set1)) * 100
percent_dataset2 = (common_count / len(set2)) * 100

print(f"\nPhần trăm câu giống nhau trong tóm tắt tự động: {percent_dataset1:.2f}%")
print(f"Phần trăm câu giống nhau trong tóm tắt tham chiếu: {percent_dataset2:.2f}%")