# Concept Drift Handling Based on Clustering in the Model Space for Class-Imbalanced Learning (CDCMS.CIL)

This repository contains the followings:
 - The MOA implementation of CDCMS.CIL with prequential accuracy weighting only (At Implementation/moa/src/main/java/moa/classifiers/meta/CDCMS_CIL.java)
 - The MOA implementation of CDCMS.CIL with prequential accuracy weighting and oversampling (OS) or undersampling (US) options (At Implementation/moa/src/main/java/moa/classifiers/meta/CDCMS_CIL_OSUS.java)
 - The MOA implementation of CDCMS.CIL with time-decay G-Mean weighting only (At Implementation/moa/src/main/java/moa/classifiers/meta/CDCMS_CIL_GMean.java)
 - The MOA implementation of CDCMS.CIL with time-decay G-Mean weighting and oversampling (OS) or undersampling (US) options (At Implementation/moa/src/main/java/moa/classifiers/meta/CDCMS_CIL_GMean_OSUS.java)
 - The supplementary material to the paper (The_Value_of_Diversity_for_Dealing_with_Concept_Drift_in_Class_Imbalanced_Data_Streams_Supplementary_Document.pdf)

## Abstract
Concept drift and class imbalance are critical challenges in real-time data stream learning. Existing ensemble methods use homogeneous diversity (models for the same concept) to tackle these challenges but often overlook heterogeneous diversity (models from different concepts), which could improve adaptation, especially with scarce minority data. This paper provides the first analysis of when and why each type of diversity is beneficial for class-imbalanced data streams. To enable this analysis, we introduce CDCMS.CIL, a novel class imbalance learning framework for leveraging heterogeneous diversity. Experiments based on 80 artificial and 9 real-world data streams show that heterogeneous diversity can significantly aid concept drift handling in highly imbalanced scenarios, while homogeneous diversity is better during stable periods. These findings provide crucial guidance for designing robust ensembles for drifting class imbalanced data streams.

#### Author
 - Chun Wai Chiu (Michael): michaelchiucw at gmail dot com
 - Leandro Minku: L dot L dot Minku at bham dot ac dot uk

#### Environment details
 - Java version: 11.0.1
 - MOA version: 2018.6.0