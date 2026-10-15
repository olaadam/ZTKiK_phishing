import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
import kagglehub

def load_and_preprocess_dataset():
    print("Pobieranie bazy danych z Kaggle")
    # Pobranie zestawu danych 'taruntiwarihp/phishing-site-urls' przy użyciu kagglehub
    path = kagglehub.dataset_download("taruntiwarihp/phishing-site-urls")
    print(f"Zbiór danych został pobrany do ścieżki: {path}")
    
    # Wczytanie pliku CSV
    import os
    csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
    csv_file_path = os.path.join(path, csv_files[0])
    
    print(f"Wczytywanie pliku: {csv_file_path}")
    df = pd.read_csv(csv_file_path)
    
    print("\nPodstawowa eksploracja danych")
    print(f"Liczba wierszy i kolumn: {df.shape}")
    print("Pierwsze 5 wierszy:")
    print(df.head())
    
    # Sprawdzenie rozkładu klas (zazwyczaj kolumny to 'URL' oraz 'Label')
    # Nazwy kolumn mogą się różnić w zależności od wersji pliku, np. ('URL', 'Type')
    print("\nRozkład etykiet w zbiorze:")
    print(df.iloc[:, 1].value_counts()) # Zakładamy, że kolumna z etykietą jest drugą kolumną
    
    # Czyszczenie danych (usunięcie duplikatów i ewentualnych pustych wierszy)
    df = df.dropna().drop_duplicates()
    
    # Przykład ekstrakcji prostych cech strukturalnych (zgodnie z planem projektu)
    print("\nEkstrakcja cech strukturalnych")
    url_column = df.columns[0]  # Pierwsza kolumna to adres URL
    label_column = df.columns[1] # Druga kolumna to etykieta (good/bad)
    
    # Przykładowe cechy numeryczne: długość URL oraz liczba kropków
    df['url_length'] = df[url_column].astype(str).apply(len)
    df['dot_count'] = df[url_column].astype(str).apply(lambda x: x.count('.'))
    
    # Podział na zmienną objaśniającą (X) i etykietę (y)
    X_urls = df[url_column].astype(str)
    X_numeric = df[['url_length', 'dot_count']]
    y = df[label_column]
    
    print("\nWektoryzacja TF-IDF dla adresów URL")
    # Używamy wektoryzacji znakowej/słownej TF-IDF
    tfidf = TfidfVectorizer(ngram_range=(1, 3), analyzer='char', max_features=5000)
    X_tfidf = tfidf.fit_transform(X_urls)
    
    # Połączenie cech tekstowych (TF-IDF) z cechami strukturalnymi (liczbowymi)
    # Konwertujemy cechy numeryczne do formatu macierzy rzadkiej, aby połączyć je z TF-IDF
    from scipy.sparse import csr_matrix
    X_numeric_sparse = csr_matrix(X_numeric.values)
    X_combined = hstack([X_tfidf, X_numeric_sparse])
    
    print(f"Kształt macierzy cech po połączeniu: {X_combined.shape}")
    
    print("\nPodział na zbiór treningowy i testowy (80/20)")
    X_train, X_test, y_train, y_test = train_test_split(
        X_combined, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Liczba próbek treningowych: {X_train.shape[0]}")
    print(f"Liczba próbek testowych: {X_test.shape[0]}")
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_preprocess_dataset()
    print("\nDane zostały pomyślnie pobrane, przetworzone i przygotowane do treningu modeli Naive Bayes / SVM!")