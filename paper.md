
# A. Problem Setup  

Our objective in this work is to estimate RR in the setup of binary exposure and outcome. Consider the population of patients described by a tuple generated independently and identically:    $(X_{i},Y_{i},T_{i})\;\;\sim\;\;P$  . Each patient    $i$   is described by medical records,    $X_{i}$   and is assigned exposure status,  $T_{i}\in\{0,1\}$  . The exposures,    $T_{i}$  , in the presented work are two classes of anti hypertensive s with one of the classes acting as reference group. The variable    $Y_{i}$   corresponds to the observed outcome—cancer—in our proposed investigations. In a ﬁxed amount of “follow-up” time after hypothetical treatment,  $T_{i}=0$  , outco  is notated as    $Y_{i}^{T=0}$  , and similarly for treatment,  $T_{i}=1,\,Y_{i}^{T=1}$   = . These two outcome variables are known as the potential outcomes under the Neyman–Rubin potential outcomes framework [40].  

With this, the RR is deﬁned as  

$$
R R=\frac{\mathbb{E}\big[Y^{T=1}\big]}{\mathbb{E}\big[Y^{T=0}\big]}.
$$  

As is fundamental to the problem of causal inference, only one of the two outcomes are observed, so equation (1) cannot directly be computed. However, with the following standard assumptions, the exposure effect is ident i able and RR is estimable.  

1. Consistency:  The potential outcome for T is the observed outcome if the given exposure was indeed T.
2. Positivity:  For all X, there is a nonzero probability of being assigned any exposure status, $T_{i}\in\{0,1\}$.
3. Un confounded ness or “no Hidden Confounding”:  The potential outcomes are independent of the exposure given all con founders are adjusted for. In synthetic data experimental designs, this assumption is more securable. In reality however, this is not measurable, but with richer observational data in the form of comprehensive medical records comprising of various health indicators (e.g., diagnoses, medications, measurements data),  

![](https://cdn-mineru.openxlab.org.cn/model-mineru/prod/c964fc676a33cf7b6798f3d56200f7494f788046592973b2dfd6a19482f7507c.jpg)  
Fig. 1. Representation learning data selection pipeline. We use Clinical Practice Research Datalink (CPRD) and extract diagnoses, medications, blood pressure, smoking, region, and sex records. We homogenize codes from ICD-10 and read to one format. Unmapped read codes were kept for completeness.  

con founders can be better adjusted [41]. Under these assumptions, the causal effect is ident i able and naïve RR estimator can be deﬁned as  

$$
\hat{\psi}=\mathbb{E}\bigg[\frac{\mathbb{E}[Y|X,T=1]}{\mathbb{E}[Y|X,T=0]}\bigg].
$$  

Other more complex estimators utilizing propensity score such as CV-TMLE are also implemented in this work.  

# B. Dataset and Patient Selection  

For our investigations, we used a data cut from CPRD, which has been described previously [31]. The data entail records from 1 January 1985 up to 31 December 2015 and is linked to national administrative databases including hospit aliz at ions (Hospital Episode Statistics, or HES) and death registration (from Ofﬁce of National Statistics).  

The dataset for the investigations was restricted to patients in the database who met the following criteria:

1) registered with the general practice for at least 12 months;
2) aged  $\geq\!16$   years at registration;
3) registered with the practice considered providing “up-to-standard” data to CPRD;
4) individual data marked by CPRD to be of “acceptable” quality for research purposes (as determined by CPRD);
5) registered with a practice that provided consent for linking the data with national databases for hospitalizations and death registry.  

We extracted diagnoses, medications, blood pressure measurements, sex (male, female), region (ten regions in U.K.), and smoking status (non, previous, or current smoker). We mapped diagnoses and medication codes to a homogenized format for machine readability. This led to a dataset of 6777845 patients, which was used for general representation learning (shown in Fig. 1) for deep learning models.  

For our causal inference investigation (i.e., investigating the effect of anti hypertensive on incident cancer), a dataset containing ﬁve sub populations had to be selected—one for each class of anti hypertensive s: ACEIs, diuretics, calcium channel blockers (CCBs), beta blockers (BBs), and angiotensin II receptor blockers (ARBs). Patients were selected in one of these groups based on ﬁrst class of anti hypertensive medications recorded before 2009 and if free of cancer report before this ﬁrst prescription; the year 2009 was chosen conveniently to have sufﬁcient “follow-up” time for the occurrence of potential cancers. The date of this ﬁrst prescription was deﬁned as “baseline” (a date between 1985 and 31 December 2008). Patients were then followed up from baseline until cancer diagnosis (including cancer diagnoses as cause of death) or end of ﬁve-year follow-up period. The learning period included the entire patients’ medical records up to a random point between six and 12 months before baseline; this is to account for any potential inaccuracies in timing of prescription (or decision to prescribe) and to avoid possibility of anti hypertensive prescription itself inﬂuencing the model training. “CPRD Product codes” are used for identifying classes of anti hypertensive s and the set of codes were obtained from a dataset published by University of Bristol [42]. Codes for cancer are found in Table II and derived from clinically established publication of codes [43].  

# C. Semi-Synthetic Data Derivation

Data generation of sequential, temporal variables is a difﬁcult task, and currently, there is no medically validated method of generating realistic EHR medical history. Thus, we utilized the existing medical history in observational data to exclusively simulate binary factual and counter factual outcomes.  

Inspired by other semi-synthetic data simulations [37], [44], intuitively, we ﬁrst modeled the association between a medical history variable    $Z_{i}$   (e.g., some diagnosis/medication) and  $T_{i}$  ith the empirical propensity in the   $\lambda_{i}=$   $P(T_{i}=1\mid Z_{i})$   =  | . If associated with an  ure (  $(\lambda_{i}~\neq~0.5)$  ) , we generated the potential outcomes,  $Y_{i}^{T=1}$  and  $Y_{i}^{T=0}$  as a function of    $\lambda_{i}$   and exposure    $T_{i}=1$   and    $T_{i}=0$  , respectively. In this way, semi-synthetic outcomes arose from an association between    $Z_{i}$   and exposure and    $Z_{i}$   and the outcome. Thus, the relationship between exposure and outcome is confounded by    $Z_{i}$  . While the empirical RR—the proportion of the outcome in one exposure group divided by the same in the other—would yield confounded causal conclusions, effectively adjusting for the confounder variable,  $Z_{i}$  , would yield identiﬁable (see Appendix Section A) causal association between exposure and outcome.  

In addition, to test model adjustment potential in situations of varying confounding intensity, we weighted the contribution of the confounding with a    $\beta$   factor: the greater the  $\beta$   implies the greater the confounding. More details of the semi-synthetic data generative process and functions modeled are given in Appendix Section A.  

In our work, we present investigations in semi-synthetic data utilizing two forms of con founders: persisting and transient confounding. We deﬁne persisting confounding as con founders that are assigned at birth and persist through one’s life course; e.g., ethnicity, sex, genes, and other variables assigned at birth that associate with variables later in age. We deﬁne transient confounding as con founders that manifest at a point or period of one’s life effecting events downstream in time; e.g., disease diagnoses, age itself, prescriptions, and other variables not assigned at birth. These two distinctions of confounding are presented in this work because they naturally capture prevalent forms of confounding seen in population health databases [45].  

From our observational dataset, we investigated two exposure groups—ACEIs and diuretics and noticed female sex was associated with the diuretics exposure status and thus, chose it to be a persistent confounder and generated conditional outcomes. For another pair of exposures, i.e., ARBs and CCBs, we identiﬁed association of incidence of at least one of heart failure, hypertension, ischemic heart disease, and diabetes mellitus to CCBs. Thus, we named occurrence of at least one of these diseases as “car dio metabolic diseases” and utilize it as a transient confounder for the second set of semi-synthetic data experiments. We set low, medium, and high confounding intensity for experiments with sex and car dio metabolic disease as confounder (  $\beta$   values: [1, 5, 10] and [25, 50, 75], respectively) totaling six experiments on semi-synthetic data. In sum, with this confounding generation method, model confounding adjustment ability will be tested with two forms of confounding at various degrees of intensity (  $\beta$   values).  

On the semi-synthetic dataset with highest intensity of cardio metabolic disease confounding, we additionally conducted ﬁnite-sample causal estimation experiments. Since estimation in limited sample settings is known to be unstable in many cases (e.g., for inverse probability weighted estimators) despite asymptotic guarantees [15], we wished to assess our model for ﬁnite-sample estimation ability. And, we spec i call y set the confounding to the highest intensity level   $\beta\,=\,75)$   because we wished to investigate how the model performs in estimation of RR in situations of high confounding. We investigated the ﬁnite-sample estimation ability of our proposed model and other deep learning models by applying the models on random subsamples of this dataset:   $2.5\%$  ,   $5\%$  ,   $10\%$  ,  $25\%$  ,   $50\%$  , and ﬁnally, the entire dataset.  

# D. Proposed Model Development  

Our model, T-BEHRT, utilizes a modiﬁed feature BEHRT extractor to capture both static and temporal medical history variables and captures initial estimates of RR. BEHRT is a state-of-the-art transformer model for EHR data. By using contextual i zed embeddings to represent longitudinal clinical encounters (e.g., diagnoses/prescriptions) and time of medical visit—both relative in terms of age/visit number and absolute in terms of calendar year—and multihead self-attention for feature extraction, BEHRT has demonstrated the state-of-theart performances in various EHR-based tasks [5], [16]. After predicting propensity score and conditional outcomes, we use CV-TMLE to correct for bias in initial RR estimate and compute corrected RR (see Fig. 2).  

Intuitively, T-BEHRT ﬁrst extracts latent EHR features from static covariates and ﬁxed sub sequences of medical history with BEHRT. Second, the model predicts propensity of exposure and conditional outcome using these learned features. Third, by additionally conducting auxiliary unsupervised learning, the model trains on reconstruction of both static and temporal data with two-part masked EHR modeling (MEM).  

The propensity prediction model is modeled as 1-hidden layer multilayer perceptron (MLP) and for each conditional outcome, we use a 2-hidden layer MLP with exponential linear unit (ELU) activation.  

With patient data tuple    $(X_{i},Y_{i},T_{i})$   as described in Appendix Section A, parameters    $\theta$  , propensity prediction head    $g(X_{i})$  , and conditional outcome prediction heads,    $H(X_{i},T_{i})$   for input  
![](https://cdn-mineru.openxlab.org.cn/model-mineru/prod/2417e1bd8711389bc3c02141bb1238aa5d6dc99a0f603a6a8c0f9a2bcd41ba09.jpg) 

Fig. 2. Targeted BEHRT and embedding structure. (a) Above, the model is shown. Generally, an input    $x$   (static and temporal variables) is fed to a feature extractor, which outputs a dense latent state (for EHR modeling, this feature extractor is BEHRT). The output of the ﬁnal layer of the BEHRT feature extractor is fed to the MEM prediction head to predict any masked encounters.   $\mathrm{T}_{N+1}$   token state is fed to a variation al auto encoder (VAE) neural network to predict masked static variables. The latent state of the ﬁrst token (T1) is fed to a pooling layer to predict propensity and conditional outcomes with multiple prediction heads with feed forward (FF) neural network layers. The loss consists of the unsupervised loss from two MEM components—temporal (temp) and static (static) unsupervised data training— and the supervised loss of the propensity and factual outcomes. (b) Below, the embedding structure for modeling rich EHR data is shown. Clinical encounters time stamped by age/year/position (visit number) are converted into vector representations and fed to model as temporal variables. Static data variable embeddings: patient sex, region in U.K., and smoking status are concatenated to the temporal variable embeddings.  

$X_{i}$   and exposure    $T_{i}$   for patient  $i$  , the loss is  

$$
\begin{array}{r l}&{\hat{O}(X_{i};\theta)=\mathrm{CrossEnKFy}(H(X_{i},T_{i};\theta),Y_{i})}\\ &{\qquad\qquad\qquad\qquad\qquad+\operatorname{CrossEnKFy}(g(X_{i};\theta),T_{i}).}\end{array}
$$  

Next, we conduct MEM for two-part unsupervised learning: 1) temporal variable and 2) static variable modeling. The ﬁrst part—unsupervised learning on temporal data—functions similar to masked language modeling (MLM) in natural language processing [46]. In MLM, the model receives a combination of masked, replaced, and unperturbed tokens (temporal or textual data) and the task is to predict the masked or replaced encounters. We do the same but additionally enforce another constraint: when replacing encounters, we do not replace encounters with those that deﬁne the exposure or outcome— anti hypertensive s and cancer in the current set of experiments. With encounter  $j$   for patient    $i$   represen  as    $E_{i,j}\subset X_{i}$   (i.e., encounters being a subset of the input  $X_{i}$  ) , masked/replaced encounters represented as  $\tilde{E}_{i,j}$  , BEHRT feature extractor    $B$  , temporal unsupervised prediction network  $M$  , neural network parameters    $\phi_{\mathrm{MEM-Temp}}$  , we develop objective function  

$$
\begin{array}{r l}&{\widehat{\mathcal{L}_{\mathrm{MEM-Trep}}}\big(E_{i,j};\,\phi_{\mathrm{MEM-Trep}}\big)}\\ &{\quad=\sum_{j=1}^{|E_{i}|}\mathrm{CrossExp}\big(M\big(B\big(\Tilde{E}_{i,j};\,\phi_{\mathrm{MEM-Trep}}\big)\big),\,E_{i,j}\big).}\end{array}
$$  

For the second part of the MEM, static data modeling, we chose using VAE for unsupervised learning due to cumulative literature empirically demonstrating its strength in representation learning in addition to the utilization of VAE structures in other causal deep learning models such as CEVAE [35]. We model static categorical variables: region, smoking status at baseline, and sex; the three variables are embedded in high-dimensional embeddings (embedding dimensions for each variable are hyper parameters of the T-BEHRT model) and mapped (via 1-layer MLP) to the size of the encounter (temporal) embeddings and, ﬁnally, concatenated to the encounter embeddings. Thus, the BEHRT model functions as feature extractor for static/temporal variables and encoder for the VAE (see Fig. 2). The temporal variables interact with the static variables through the multihead selfattention mechanism of the BEHRT architecture [31]. For training the VAE, similar to the temporal modeling, we mask some variables as input and use a variable-speciﬁc decoder to decode the variable (if masked). Spec i call y, for static variable  $X_{i,\upsilon}$   of a total of    $V$   static v    $i$  ,    $q_{\phi_{\mathrm{Enc}}}(Z_{i}\mid X_{i})$  representing the encoder, and  $p_{\phi_{\mathrm{Dec}}}(X_{i,\upsilon}\mid Z_{i})$   | e presenting the multivariate Bernoulli decoder for variable  v , and the VAE loss is  

$$
\begin{array}{r l}&{\mathcal{L}_{\mathrm{MEM-Stark}}(x_{i};\phi_{\mathrm{Enc}},\phi_{\mathrm{Dec}})}\\ &{\quad=\displaystyle\sum_{v=1}^{V}\sum_{i=1}^{n}\log p_{\phi_{\mathrm{Dec}}}\big(X_{i,v}\mid Z_{i}\big)}\\ &{\quad\quad-\displaystyle\sum_{i=1}^{n}D_{K L}\big(q_{\phi_{\mathrm{Enc}}}(Z_{i}\mid X_{i})||p_{\phi_{\mathrm{Dec}}}(Z_{i})\big).}\end{array}
$$  

The complete objective function to be minimized is the summation of (3), (4), and (5) as shown in the following equation :  

$$
\begin{array}{r l}&{\hat{\theta},\hat{\varepsilon},\hat{\phi}_{\mathrm{Enc}},\hat{\phi}_{\mathrm{Dec}},\hat{\phi}_{\mathrm{MEM-Term}}}\\ &{\quad=\underset{\theta,\varepsilon,\phi_{\mathrm{Enc}},\phi_{\mathrm{Dec}},\phi_{U}}{\mathrm{argmin}}\displaystyle\sum_{i=1}^{n}\hat{O}\left(X_{i};\theta\right)}\\ &{\quad\quad+\delta\Big(\mathcal{L}\underset{\mathrm{MEM-Term}}{\mathrm{num-Term}}\big(E_{i,j};\phi_{\mathrm{MEM-Term}}\big)}\\ &{\quad\quad+\mathcal{L}\underset{\mathrm{MEM-Sigma}}{\mathrm{MEM-Sigma}}\left(X_{i};\phi_{\mathrm{Enc}},\phi_{\mathrm{Dec}}\right)\Big).}\end{array}
$$  

With hyper parameter  $\delta$   for weighting the contribution of the unsupervised MEM loss terms.  

# E. Feature Selection and Preprocessing  

The modalities of CPRD considered for deep learning modeling were sex, region, diagnoses from both primary and secondary care, medications, systolic blood pressure (BP) measurements, and smoking status.  

We mapped read codes from primary care and ICD-10 codes from secondary care to 1471 unique ICD-10 diagnostic codes [47], [48] to harmonize disease codes in the dataset; unmapped codes were included for completion. Furthermore, we mapped medication codes to 426 codes in the British National Formulary (BNF) [49] coding format. Since systolic BP is a continuous variable and our feature extractor requires disc ret i zed elements (see BEHRT feature extraction in Appendix Section D), systolic BP measurements (in mm  $\mathrm{Hg},$   were grouped into 16 categories based on pre spec i ed boundaries ([90–116], (116,121], (121,126],  . . . ,  (181,186],  $>\!186_{,}$  ). Furthermore, we utilized calendar year, age (months), and relative position (visit number) for the sequential/temporal modalities. Each patient    $p$   had    $n_{p}$   encounters, or instances of modalities: diagnoses, medications, and systolic BP measurements. Smoking status at baseline (non, previous, or current smoker), region (ten regions in U.K.), and sex (male, female) were static variables included in modeling.

# F. Benchmarks and Causal Estimation

Before pursuing the causal investigations with deep learning modeling, we pretrained contextual i zed EHR embeddings and network weights through MEM on the pre training dataset. This MEM task generally trains weights on all patients in CPRD before progressing to causal modeling (6777845 patients in Fig. 1).  

For semi-synthetic investigations, we implemented statistical and deep learning models to serve as benchmarked comparison models for causal inference. The benchmarks include Bayesian additive regression trees (BART) [50], LR and L1/L2 regular iz ation variants, and LR with TMLE [51]. We chose the covariates for these models to be baseline age, smoking status, sex, region, incidence of 33 curated disease groups, and additionally prescription of four additional medications groups. While inclusion of baseline variables in epi de mio logical observational studies is standard practice, we spec i call y include the disease/medication groups to enable a fairer comparison to deep learning modeling. Furthermore, diagnoses and medications are known to be con founders in observational studies, so adjustment of these variables is important for causal estimation. To ensure that the diagnoses and medication groups are medically valid clusters of diseases and medications, respectively, we utilized groups compiled by past medical research [42], [43]. A deeper explication is given in Appendix Section C.  

To serve as deep learning benchmarks, we implemented staple deep learning models for average causal effect: TARNET, TARNET    $^+$   MEM (i.e., with unsupervised MEM component), and Dragonnet with BEHRT feature extractor and the embedding format presented in Fig. 2(a). We initialized these models with pretrained weights. After implementing and evaluating benchmarks, we implemented T-BEHRT with pretrained network weights where applicable and pursue modeling of semi-synthetic data investigations.  

For the semi-synthetic data experiments, we did not feed variables denoting car dio metabolic disease and sex, respectively, as input; we wish the statistical and deep learning models to infer confounding from remaining input variables. In routine clinical data, the observational studies would often not have access to all confounding variables—thus, important to test models’ ability to adjust for confounding given limited input variables.  

For all investigations, we conducted experiments with ﬁve-fold cross validation causal estimation. We calculated RR on the test dataset for each fold as advised by Cher no zhukov  et al.  [23] and compute  $95\%$   CIs over the ﬁve folds. We computed RR deﬁned by naïve estimator on a ﬁnite sample:  $\hat{\psi}~=~\mathbb{E}[(\mathbb{E}[H(X,1)]/\mathbb{E}[H(X,0)])]$  = [ [ ] [ ] ]  for TARNET, TARNET-MEM, LR (and L1/L2 regular iz ation variants), and BART. For T-BEHRT, we use the CV-TMLE method for the estimation of RR. For Dragonnet, we implement the model with the CV-TMLE estimator in order to directly compare our model with this benchmark model. In addition, we also implement the Dragonnet model with the naïve estimator (i.e., the original model without post-hoc estimator). For more information on the CV-TMLE method, advantages over TMLE, and implementation, please refer to Appendix Section B. For models that utilized predicted propensity scores, we conducted propensity score trimming and exclude patients with predicted propensity score greater than 0.97 and less than 0.03 [52] before pursuing RR calculation.  

We identiﬁed the superior model by identifying the model with least sum absolute error (SAE) over the three  $\beta$   values for each confounding experiment. We give the standard error (SE) for the SAE; this was calculated using additive propagation of error [53]. For deep learning models, we also demonstrate change of SAE as modules are removed from our proposed model.  

# G. Implementation  

We developed all statistical and deep learning models on python. The deep learning models were implemented with Pytorch [54]. Hyper parameters for the BEHRT feature extractor are found in Table III. For training all deep learning models, we used the Adam optimizer [55] with exponential decay scheduler (decay rate  $=0.95$  ) to ensure training convergence. For TARNET-MEM and T-BEHRT, we pretrained ﬁve epochs on exclusively the MEM task before initiating joint MEM-causal task training.  

After ﬁtting deep learning and statistical models, in order to derive estimates for RR estimation, we conducted the evaluation of the model on the test fold of the dataset using standard  $\mathrm{g}.$  -computation methods [1]. For all patients in the test set, we ﬁrst derived risk estimates [e.g., estimation  $P(Y\mid X,T=0)]$  ]  patients as if they were all assigned  $T\,=\,0$   =  0, and similarly, derived estimates [e.g. ation of  $P(Y\mid X,T=1)]$   as if hey were all assigned  $T=1$   =  1. In this way, the RR estimate,   $\hat{\psi}$  , can be derived as a function of these two quantities  

$$
\hat{\psi}=\mathbb{E}\bigg[\frac{\mathbb{E}[H(X,T=1)]}{\mathbb{E}[H(X,T=0)]}\bigg].
$$  

LR (and regular iz ation variants), BART, TMLE, and CV-TMLE were implemented in python. The code was inspired by past works utilizing TMLE [3]. To ﬁt the nuisance parameter for the TMLE estimate update step, Nelder-mead optimization was utilized [6], [56]. For deep learning models implemented with CV-TMLE, the naïve estimator (7) was not used; rather, the CV-TMLE estimator was implemented utilizing conditional outcome predictions,    $H(X,T\ \ =\ \ 1)$  = ),  $H(X,T=0)$  , and propensity score prediction,    $g(X)$  .  
TABLE I P OPULATION  S TATISTICS 
![](https://cdn-mineru.openxlab.org.cn/model-mineru/prod/6a884ac483f51fb0ece982c572c3b70cb4aac3768aa3a6e7cb77fb536a84d301.jpg)  
 $\%$  