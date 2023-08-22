import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 900)

'''
CUST_ID : Kredi Kartı hamilinin kimliği (Kategorik)
BALANCE : Alım yapmak için hesabında kalan bakiye
BALANCE_FREQUENCY : Bakiyenin ne sıklıkla güncellendiği,
                    0 ile 1 arasında puan (1 = sık güncellenir, 0 = sık güncellenmez)
PURCHASES : Hesaptan yapılan satın alma miktarı
ONEOFF_PURCHASES : Tek seferde yapılan maksimum satın alma tutarı
INSTALLMENTS_PURCHASES : Taksitle yapılan satın alma tutarı
CASH_ADVANCE : Kullanıcı tarafından verilen peşin nakit
PURCHASES_FREQUENCY : Satın almaların ne sıklıkta yapıldığı, 
                      0 ile 1 arasında puan (1 = sıklıkla satın alınır, 0 = sık satın alınmaz)
ONEOFFPURCHASEFRQUENCY : Tek seferde satın almaların gerçekleşme sıklığı 
                        (1 = sık satın alınır, 0 = sık satın alınmaz)
PURCHASESINSTALLMENTSFREQUENCY : Taksitli satın almaların ne sıklıkla yapıldığı 
                                (1 = sıklıkla yapılır, 0 = sıklıkla yapılmaz)
CASHADVANCEFREQUENCY : Nakit avansın ne sıklıkta ödendiği
CASHADVANCETRX : "Vadeli Nakit" ile Yapılan İşlem Sayısı (Number of Transactions made with "Cash in Advanced")
PURCHASES_TRX : Yapılan satın alma işlemi sayısı
CREDIT_LIMIT : Kullanıcı için Kredi Kartı Limiti
PAYMENTS : Kullanıcı tarafından yapılan Ödeme Tutarı
MINIMUM_PAYMENTS : Kullanıcı tarafından yapılan minimum ödeme tutarı
PRCFULLPAYMENT : Kullanıcı tarafından ödenen tam ödeme yüzdesi
TENURE : Kullanıcı için kredi kartı hizmetinin kullanım süresi

'''


def read_file():
    dataset = pd.read_csv('credit_card_dataset.csv')
    return dataset


def data_preprocessing(dataset):
    # data preprocessing
    # check for missing values
    print(dataset.isnull().sum())

    # handling missing values
    dataset['CREDIT_LIMIT'].fillna(dataset['CREDIT_LIMIT'].mean(), inplace=True)  # 1 record
    dataset['MINIMUM_PAYMENTS'].fillna(dataset['MINIMUM_PAYMENTS'].mean(), inplace=True)  # 313 record

    # check data statistics
    print(dataset.describe().T)

    return dataset


def apply_elbow_method(dataset):
    X = dataset.values[:, 1:]
    scaled_x = StandardScaler().fit_transform(X)

    n_clusters = 15
    sum_of_squared_distances = []
    for i in range(1, n_clusters + 1):
        k_mean = KMeans(i)
        k_mean.fit(scaled_x)
        sum_of_squared_distances.append(k_mean.inertia_)

    '''
    Inertia measures how well a dataset was clustered by K-Means. 
    It is calculated by measuring the distance between each data point and its centroid, 
    squaring this distance, and summing these squares across one cluster. 
    A good model is one with low inertia AND a low number of clusters ( K ).
    '''

    plt.plot(sum_of_squared_distances, 'bx-')
    plt.xlabel('K Cluster')
    plt.ylabel('Sum of Squared Distances')
    plt.title(f'The Elbow Method for Optimal K - {n_clusters} Iteration')
    plt.show()


if __name__ == '__main__':
    dataset = read_file()

    # data preprocessing
    dataset = data_preprocessing(dataset)

    # apply elbow method
    apply_elbow_method(dataset)

    # apply K-Means
    X = dataset.values[:, 1:]
    scaled_dataset = StandardScaler().fit_transform(X)

    # according to elbow method, optimal cluster number is 3
    clustered_data = KMeans(3).fit(scaled_dataset)
    dataset['CLUSTER_ID'] = clustered_data.labels_
    # center_points = clustered_data.cluster_centers_

    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=dataset, x='PURCHASES', y='CREDIT_LIMIT', hue='CLUSTER_ID')
    plt.title('Distribution of clusters based on purchases and credit limit')
    plt.show()

    # apply PCA
    pca = PCA(n_components=2)
    reduced_x = pd.DataFrame(data=pca.fit_transform(X), columns=['PCA1', 'PCA2'])

    # Reduced Features
    # reduced_x.head()
    print(reduced_x.head())

    centers = pca.transform(clustered_data.cluster_centers_)
    print(centers)

    # scatter plot
    plt.scatter(reduced_x['PCA1'], reduced_x['PCA2'], c=clustered_data.labels_)
    plt.scatter(centers[:, 0], centers[:, 1], marker='x', s=100, c='red')
    plt.xlabel('PCA1')
    plt.ylabel('PCA2')
    plt.title('Credit Card Cluster')
    plt.show()

    updated_dataset = dataset.drop(['CUST_ID', 'CLUSTER_ID'], axis=1)
    component_df = pd.DataFrame(pca.components_, index=['PCA1', 'PCA2'], columns=updated_dataset.columns)
    sns.heatmap(component_df)
    plt.show()
