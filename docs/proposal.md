_TL;DR Can ML predict coral bleaching from temperature time series?_


### **Deliverable**
**Compare performance** of **traditional** threshold-based approaches (as outlined by previous papers) against a **machine learning** model doing time series classification (e.g. [LSTM](https://colah.github.io/posts/2015-08-Understanding-LSTMs/), [1D CNN](https://medium.com/@abhishekjainindore24/understanding-the-1d-convolutional-layer-in-deep-learning-7a4cb994c981)) for **prediction of bleaching occurrence** (a supervised classification task). If results look promising and contributors are keen, a paper!

### **Datasets**
Semi-processed datasets from the original papers are available [here](https://zenodo.org/records/11024073). These are:
1. Daily SST data from 1981-2019 for every reef site in analysis (.mat) from 5x5km satellite
2. Train-test splits as used in the paper
3. Regional subdivisions to assess regional performance (.xlsx)
4. Two .xlsx files containing bleaching information (~35000 data points)

### **Key considerations**
1. Potentially significant class imbalance
2. Some data wrangling necessary (perhaps also generating pseudo absence points) for bleaching occurrence data
3. DHWs aren't great: there's more to bleaching than heat. There may be insufficient signal for ML: but this in and of itself would be interesting.
4. Coral organisms adapt to heat. While they do this very gradually, there will be some degree of non-linearity in the bleaching response. Not to mention the fact that, if a coral gets so stressed that it dies it won't be alive to bleach in subsequent heatwave events...
