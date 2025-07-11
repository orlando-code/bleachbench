<img width="1000" height="426" alt="Image" src="https://github.com/user-attachments/assets/f4b74f3a-8ccf-4659-97db-f70020548c97" />

Image credit: [Great Barrier Reef Foundation](https://www.barrierreef.org/news/explainers/what-is-coral-bleaching)


### **Motivation**
Coral reefs are complex systems of animal-plant symbiosis on which millions of people rely for food, protection from coastal storms, and income. Shallow-water coral species – and the biodiversity they support – are threatened with functional extinction over the coming decades due to changing oceanographic conditions driven by anthropogenic greenhouse gas emissions.

Rising sea temperatures are leading to more frequent and extreme **bleaching** events. Bleaching is triggered by sustained temperatures above a certain threshold. This varies by species, location, depth etc. but people often try to predict it using the **[Degree Heating Week](https://coralreefwatch.noaa.gov/product/5km/tutorial/crw10a_dhw_product.php#:~:text=The%20DHW%20shows%20how%20much,or%20exceeds%20the%20bleaching%20threshold.) (DHW)** heuristic: the number of weeks for which temperatures remain greater than the mean monthly climatology. 

DHWs are the state-of-the-art – or at least the most [widely-used](https://coralreefwatch.noaa.gov/product/5km/index_5km_dhw.php) – in predicting coral bleaching. However, they are fairly heuristic and don't always perform well. I'd like to explore whether some **machine learning** methods could do a better job of finding a function mapping sea surface temperature time series to the occurrence of bleaching...

### **Previous work**
[Lachs et al. (2021)](https://www.mdpi.com/2072-4292/13/14/2677) fine-tuned traditional DHW methods to optimise AUC for bleaching detection. A similar approach was taken by [Whitaker and DeCarlo (2023)](https://link.springer.com/article/10.1007/s00338-024-02512-w).

Various other papers of differing quality have thrown ML algorithms at selections of climatologies and additional variables (e.g. pH, surface winds) to try and predict bleaching e.g. [Panja et al. (2024)](https://www.sciencedirect.com/science/article/pii/S0048969724031498), [Boonman et al. (2022)](https://www.mdpi.com/2071-1050/14/10/6161), [Maulidina et al. (2024)](https://ieeexplore.ieee.org/document/10862345).

