# IEEE-CIS Fraud Detection

## კონკურსის მიმოხილვა 
IEEE-CIS Fraud Detection კონკურსის მიზანი ონლაინ ტრანზაქციებში თაღლითობის აღმოჩენაა. 
ჩვენი დავალებაა, თაღლითობის ალბათობა დავთვალოთ (isFraud). გვაქვს ორი ტიპის მონაცემი:
ტრანზაქციასა და იდენტობასთან დაკავშირებით. ტრანზაქციის მონაცემებია, მაგალითად, თანხის რაოდენობა, 
გადარიცხვის დრო და სხვ. იდენტობასთან დაკავშირებული მონაცემებია, მაგალითად, მოწყობილობის (რომლიდანაც 
ტრანზაქცია განხორციელდა) ტიპი, ბრაუზერი, ფოსტა და სხვ. გარდა ამისა, არის ანონიმური მონაცემები, 
რომელთა რეალური მნიშვნელობა უცნობია. მთავარი პრობლემა ამ კონკურსში, ჩემი აზრით, ისაა, რომ 
კლასებს შორის დიდი განსხვავება ანუ დისბალანსია- თაღლითური ტრანზაქციები ძალიან იშვიათია (~3%). 
accuracy, შესაბამისად, კარგი მეტრიკა არ იქნებოდა (ყველა მონაცემზე ვიტყოდით, რომ არაა თაღლითური 
და 97%-იანი accuracy-ის მაჩვენებელი გვექნებოდა). სწორედ ამიტომ კონკურსის მეტრიკად roc-auc არის 
არჩეული, რომელიც precision/recall ბალანსზეა დამოკიდებული და ზუსტად მიემართება აღნიშნულ კონკურსს.

## რეპოზიტორიის სტრუქტურა

```
project/
- exploratory data analysis (EDA)
- model-experiment-logisticregression.ipynb
- model-experiment-randomforest.ipynb
- model-experiment-xgboost.ipynb
- model-experiment-modelinference.ipynb
- README.md
```

1. EDA: kaggle-ის ინფუთიდან ჩამოვტვირთე train_transaction.csv და train_identity.csv, შემდეგ 
კი გავაერთიანე ეს მონაცემები TransactionID-ის მიხედვით. რადგან ძალიან ბევრი მონაცემია, 
მეხსიერების ოპტიმიზაცია გადავწყვიტე, რათა უფრო ჩქარა დამემუშავებინა დატასეტი. მონაცემები 
გავყავი train/validation ნაწილებად (შეფარდებით 80/20) stratified sampling-ის გამოყენებით, რათა
fraud ratio შემენარჩუნებინა. შევხედე isFraud-ის დისბალანსს. დავალაგე feature-ები nan-ების 
მიხედვით. შემდგომში ამ ინფორმაციას გამოვიყენებ გადასაყრელად იმ სვეტებისა, რომლებსაც nan-ების
მაღალი პროცენტულობა აქვთ. TransactionAmt-ის გასაანალიზებლად გამოვიყენე ამ სვეტის მნიშვნელობათა
ლოგარითმი, რათა გამომესწორებინა skewness (არათანაბარი განაწილება). ტრანზაქციების უმეტესობა 
მცირეა, თუმცა რიგ შემთხვევებში ძალიან დიდია, რაც გადახრილ განაწილებას ქმნის, log transform 
კი განაწილებას აბალანსებს. ასევე გავაანალიზე კატეგორიული ცვლადები, მაგალითად, productCD. 
C კატეგორია, მაგალითად, თაღლითობასთან უფრო ხშირად ასოცირდება. გამოვიკვლიე ფოსტის დომენებიც. 
TransactionDT-ზე დაყრდნობით შემოვიღე რამდენიმე ახალი ცვლადი- საათი და კვირის დღე, რათა
გამეანალიზებინა თაღლითობის დამოკიდებულება დროზე. ასევე შემოვიღე ათწილადის აღმნიშვნელი სვეტი. ამ feature-ის მიზანია გამოვლენა იმ მცირე რიცხვითი პატერნებისა, 
რომლებიც შესაძლოა ასოცირებული იყოს თაღლითობასთან.
2. model-experiment-logisticregression.ipynb- ლოჯისტიკური რეგრესიის მოდელის დატრენინგება. 
3. model-experiment-randomforest.ipynb- random forest-ის მოდელის დატრენინგება.
4. model-experiment-xgboost.ipynb- xgboost-ის მოდელის დატრენინგება. 
4. model-experiment-modelinference.ipynb- საუკეთესო მოდელის ჩამოტვირთვა და kaggle-ის submission-ის
დაგენერირება. 

## Data Cleaning and Feature Selection
სამივე მოდელში ერთნაირი data cleaning-ია. MissingValueDropper-ის გამოყენებით გადავყარე სვეტები, 
რომლებსაც ძალიან ბევრი გამოტოვებული მნიშვნელობა ჰქონდათ. threshold-ად თავდაპირველად 0.9 ავიღე, 
თუმცა უკეთესი შედეგი სამივე მოდელზე 0.95-მა აჩვენა. 

Vxxx სვეტები შევამცირე VColumnReducer-ის გამოყენებით. V სვეტების მნიშვნელობაზე ინფორმაცია 
არ გვაქვს, თუმცა დავადგინე, რომ მათი დიდი ნაწილი ძლიერ კორელირებულია. ამ სვეტების შემცირებამ 
არა მხოლოდ უკეთესი შედეგი დადო, არამედ შეამცირა training დრო და მონაცემების განზომილება. 

nan-ები რიცხვითი ცვლადების შემთხვევაში შევცვალე მედიანით, ხოლო კატეგორიული ცვლადების შემთხვევაში კონსტანტით
("missing") და ordinalEncoder-ით xgboost-ისა და randomForest-ის შემთხვევაში, ხოლო woeEncoder-ით
ლოჯისტიკურ რეგრესიაში. ამ უკანასკნელის შემთხვევაში გამოვიყენე Scaling- წრფივი მოდელების ოპტიმიზაციისთვის 
ის საჭიროა, რადგან, როგორც ვიცით, ლოჯისტიკური რეგრესია გულისხმობს წონების ვექტორისა და მონაცემების წრფივ 
კომბინაციას და თუ რომელიმე მონაცემი მცირეა, ის პრაქტიკულად იგნორირდება, არადა შესაძლოა, სასარგებლო ინფორმაციას
ინახავდეს- სწორედ ამიტომ ვიყენებთ scaling-ს. სამაგიეროდ scaling არაა საჭირო ხის მოდელებში, რომლებიც split-ებს
threshold-ების მიხედვით აკეთებს. 
woeEncoder თავის თავში ინახავს კავშირს სვეტსა და target-ს
შორის, ამიტომ წრფივ მოდელში საჭიროა. ხის მოდელები split-ებზე რაკია დამოკიდებულია, woeEncoder-ზე 
უკეთესი აქ ordinalEncoder-ია, რომელიც ordering-ისთვისაა კარგი (რასაც ხეები იყენებენ). 

ასევე გამოვიყენე VarianceThreshold(), რომლითაც მოვაშორე ისეთი სვეტები, რომლებშიც თითოეულ მონაცემზე ერთი და
იგივე მნიშვნელობა აქვთ და, შესაბამისად, არაინფორმატიულია. 

# Feature Engineering
შემოვიღე შემდეგი ინფორმატიული ცვლადები:
1. TransactionAmt_log- ტრანზაქციის თანხის ლოგარითმი, რაც განაწილების ნორმალიზაციისთვის უკეთესია (მცირდება skewness);
2. TransactionAmt_decimal- ტრანზაქციის თანხის ათწილადი ნაწილი;
3. Transaction_day- TransactionDT-ის მიხედვით შემოვიღე ტრანზაქციის დღე;
4. Transaction_hour- TransactionDT-ის მიხედვით შემოვიღე ტრანზაქციის საათი. 

# Training 
ვალიდაციის სტრატეგიად ავირჩიე დაყოფა დროის მიხედვით (time-based split): 
ტრანზაქციების პირველი 80% გამოიყენებოდა training-ისთვის, 
ხოლო ბოლო 20% — validation-ისთვის. 
ეს მნიშვნელოვანია იმიტომ, რომ მონაცემები წარმოადგენს დროში თანმიმდევრულ მოვლენებს და არა ერთმანეთისგან დამოუკიდებელ შემთხვევით ჩანაწერებს.
შემთხვევითი დაყოფის შემთხვევაში წარსული და მომავალი მონაცემები ერთმანეთში ირევა, რაც იწვევს დროითი data leakage-ს:
მოდელს შეიძლება “გაუცნობიერებლად” შეხვდეს მომავლის ინფორმაცია. 
validation შედეგები ხდება არარეალისტურად მაღალი.

fraud detection სისტემაში ყოველთვის ვმუშაობთ მომავალ ტრანზაქციებზე წარსული მონაცემების საფუძველზე.
Kaggle-ის test set-შიც უფრო გვიანდელი ტრანზაქციებია, ცხადია. 

# 1. Logistic Regression:
დავტესტე რამდენიმე კონფიგურაცია (lbfgs, liblinear, saga): ამ სამს შორის საუკეთესო შედეგი lbfgs-მ დადო, ამიტომ
ის კიდევ უფრო დეტალურად გავტესტე.

| Run | SMOTE | C | Solver | Penalty | Train AUC | Validation AUC | AUC Gap | Fit Time |
|-----|------:|--:|--------|----------|----------:|---------------:|--------:|---------:|
| LR_C_0.1_lbfgs_l2 | False | 0.1 | lbfgs | l2 | 0.861937 | 0.826968 | 0.034969 | 23.67s |
| LR_C_1.0_lbfgs_l2 | False | 1.0 | lbfgs | l2 | 0.862937 | **0.831581** | 0.031356 | 35.15s |
| LR_C_10.0_lbfgs_l2 | False | 10.0 | lbfgs | l2 | 0.862817 | 0.831529 | 0.031288 | 38.36s |
| LR_C_0.1_lbfgs_l2 | True | 0.1 | lbfgs | l2 | 0.867904 | 0.830537 | 0.037368 | 77.37s |
| LR_C_1.0_lbfgs_l2 | True | 1.0 | lbfgs | l2 | 0.868005 | 0.830776 | 0.037229 | 88.04s |
| LR_C_10.0_lbfgs_l2 | True | 10.0 | lbfgs | l2 | 0.868028 | 0.830914 | 0.037114 | 85.18s |

საუკეთესო მოდელი აღმოჩნდა LR_C_1.0_lbfgs_l2_smote_False.

კლასების დისბალანსის გათვალისწინებით ვცადე smote. როგორც ლექციაზე ვთქვით, 
oversampling ზოგჯერ ეხმარება მოდელს minority კლასის უკეთ შესწავლაში.
თუმცა ამ ექსპერიმენტში SMOTE-მა Logistic Regression-ის შედეგები არ გააუმჯობესა.
SMOTE-ის გარეშე საუკეთესო მოდელმა მიიღო validation AUC = `0.831581`, ხოლო SMOTE-ით საუკეთესო მოდელმა მიიღო validation AUC = `0.830914`.
ასევე SMOTE მნიშვნელოვნად ზრდიდა training დროს:

SMOTE-ის გარეშე: დაახლოებით `24–38` წამი
SMOTE-ით: დაახლოებით `77–88` წამი

threshold `0.5`-ზე SMOTE-მა გამოიწვია precision-ის მკვეთრი შემცირება:

| მოდელი            | SMOTE | Validation Precision @ 0.5 | Validation Recall @ 0.5 | Validation F1 @ 0.5 |
| ----------------- | ----: | -------------------------: | ----------------------: | ------------------: |
| C=1.0, lbfgs, l2  | False |                   0.550447 |                0.212106 |            0.306217 |
| C=10.0, lbfgs, l2 |  True |                   0.098743 |                0.751969 |            0.174564 |

ეს აჩვენებს, რომ SMOTE-მა მოდელი აიძულა უფრო ხშირად დაეკლასიფიცირებინა ტრანზაქციები fraud-ად. 
შედეგად recall გაიზარდა, მაგრამ precision მკვეთრად დაეცა, რაც ნიშნავს ბევრ false positive-ს.

რადგან SMOTE-მა არ გააუმჯობესა validation AUC და მნიშვნელოვნად გააუარესა precision,
საბოლოო Logistic Regression მოდელისთვის ის არ ავირჩიე.

სტანდარტულად threshold არის 0.5. თუმცა fraud detection-ში ეს მნიშვნელობა ხშირად არ არის ოპტიმალური, 
რადგან dataset ძალიან არაბალანსირებულია.

საუკეთესო Logistic Regression მოდელისთვის SMOTE-ის გარეშე შედეგები threshold `0.5`-ზე იყო:

Validation Precision: `0.550447`
Validation Recall: `0.212106`
Validation F1: `0.306217`

F1-ს მიხედვით საუკეთესო threshold იყო `0.238549`. ამ threshold-ზე validation F1 გაიზარდა `0.306217`-დან `0.377613`-მდე.

threshold-ის შემცირება აუმჯობესებს precision-სა და recall-ს შორის ბალანსს.

საუკეთესო Logistic Regression მოდელს ჰქონდა შემდეგი შედეგები:

Train AUC: `0.862937`
Validation AUC: `0.831581`
AUC Gap: `0.031356`

Train და validation შედეგებს შორის სხვაობა შედარებით მცირეა, 
რაც ნიშნავს, რომ მოდელი არ არის overfitted. 
თუმცა როგორც train, ისე validation AUC შედარებით დაბალია უფრო ძლიერი tree-based მოდელებთან (მაგალითად XGBoost-თან) 
შედარებით. ეს მიუთითებს, რომ Logistic Regression-ის შემთხვევაში underfitting-ს ვაწყდებით.

ამის მთავარი მიზეზი არის ის, რომ Logistic Regression არის წრფივი მოდელი. 
Fraud detection პრობლემებში ხშირად არსებობს არაწრფივი დამოკიდებულებები ტრანზაქციის თანხას, ბარათის ინფორმაციას, 
იდენტობას, email domain-ებს, მისამართებსა და დროით ქცევას შორის. 
Logistic Regression ვერ ახერხებს ასეთი რთული ურთიერთქმედებების სრულად მოდელირებას, 
განსხვავებით tree-based მოდელებისგან.

ყველა Logistic Regression ექსპერიმენტი დავარეგისტრირე MLflow-ზე Logistic_Regression_Training ექსპერიმენტის ქვეშ.

ყოველ run-ზე ილოგებოდა:

მოდელის პარამეტრები: 
- C, 
- solver, 
- penalty, 
- max_iter,
- გამოყენებული იყო თუ არა SMOTE,
- train AUC,
- validation AUC,
- AUC gap,
- average precision,
- precision,
- recall,
- F1-score,
- საუკეთესო F1 threshold,
- confusion matrix-ის მნიშვნელობები,
- training დრო (fitting time),
- convergence warning-ების რაოდენობა.

საუკეთესო Logistic Regression კონფიგურაცია იყო:

- C = 1.0
- solver = lbfgs
- penalty = l2
- SMOTE = False

ამ მოდელმა მიიღო validation AUC = `0.831581`.

SMOTE ამ მოდელისთვის არ აღმოჩნდა სასარგებლო, რადგან არ გააუმჯობესა validation AUC, გაზარდა training დრო და მნიშვნელოვნად გააუარესა precision threshold `0.5`-ზე.

ამრიგად, Logistic Regression გამოვიყენე, როგორც baseline მოდელი. 
ის იყო შედარებით სტაბილური და არ განიცდიდა ძლიერ overfitting-ს, 
თუმცა ამოცანის სირთულესთან მიმართებით განიცდიდა underfitting-ს. 
ამიტომ ის არ შევარჩიე საბოლოო საუკეთესო არქიტექტურად.

# 2. Random Forest
დისბალანსის გათვალისწინებით ორი მიდგომა გამოვიყენე: class_weight="balanced_subsample"
(თითოეულ bootstrap sample-ში არეგულირებს კლასების წონებს, რათა minority class-ს (fraud) მეტი მნიშვნელობა მიენიჭოს.)
და SMOTE.

რამდენიმე Random Forest კონფიგურაცია დავატრენინგე. მინდოდა შემდეგი პარამეტრების ოპტიმიზაცია:
- ხეების რაოდენობა (number of trees),
- ხის სიღრმე (tree depth),
- მინიმალური samples თითო ფოთოლზე (min samples per leaf),
- feature sampling (max features),
- row sampling (max samples),
- SMOTE-ის გამოყენება.

| Run | SMOTE | Trees | Max Depth | Min Samples Leaf | Max Features | Max Samples | Train AUC | Validation AUC | AUC Gap | Fit Time |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| RF_trees_300_depth_20_smote_False | False | 300 | 20 | 8 | sqrt | 0.8 | 0.986278 | **0.892833** | 0.093445 | 197.56s |
| RF_trees_300_depth_20_smote_True | True | 300 | 20 | 8 | sqrt | 0.8 | 0.955577 | 0.883330 | 0.072247 | 706.64s |
| RF_trees_250_depth_14_smote_False | False | 250 | 14 | 10 | sqrt | 0.7 | 0.949427 | 0.883322 | 0.066105 | 124.60s |
| RF_trees_150_depth_10_smote_False | False | 150 | 10 | 20 | sqrt | 0.7 | 0.908100 | 0.871845 | 0.036255 | 61.70s |
| RF_trees_250_depth_14_smote_True | True | 250 | 14 | 10 | sqrt | 0.7 | 0.909411 | 0.864331 | 0.045081 | 450.77s |
| RF_trees_150_depth_10_smote_True | True | 150 | 10 | 20 | sqrt | 0.7 | 0.876639 | 0.848928 | 0.027710 | 214.87s |

საუკეთესო Random Forest კონფიგურაცია იყო:

RF_trees_300_depth_20_smote_False

Validation AUC = `0.892833`

საუკეთესო მოდელმა SMOTE-ის გარეშე მიიღო:

Validation AUC = `0.892833`

ხოლო საუკეთესო მოდელმა SMOTE-ით მიიღო:

Validation AUC = `0.883330`

როგორც გამოჩნდა, SMOTE-მა Random Forest-ის შემთხვევაში validation AUC არ გააუმჯობესა.

SMOTE-მა ასევე მნიშვნელოვნად გაზარდა training დრო. 300 ხის და depth=20 მოდელისთვის training დრო გაიზარდა:
- SMOTE-ის გარეშე: `197.56` წამი
- SMOTE-ით: `706.64` წამი

threshold `0.5`-ზე SMOTE-იანმა მოდელმა აჩვენა უფრო მაღალი precision და F1, მაგრამ დაბალი recall და დაბალი AUC:

| Model | SMOTE | Validation Precision @ 0.5 | Validation Recall @ 0.5 | Validation F1 @ 0.5 | Validation AUC |
|---|---:|---:|---:|---:|---:|
| 300 trees, depth 20 | False | 0.407127 | 0.489173 | 0.444395 | 0.892833 |
| 300 trees, depth 20 | True | 0.668161 | 0.366634 | 0.473467 | 0.883330 |

რადგან fraud detection არის არაბალანსირებული კლასიფიკაციის პრობლემა, AUC-ის გარდა დამატებით ვლოგავდი threshold-ზე დამოკიდებული მეტრიკებსაც.
საუკეთესო Random Forest მოდელისთვის validation შედეგები threshold `0.5`-ზე იყო:

Validation Average Precision: `0.482446`,
Validation Precision: `0.407127`,
Validation Recall: `0.489173`,
Validation F1: `0.444395`.

საუკეთესო F1-ის მიხედვით მიღებული threshold იყო:
Best F1 threshold = `0.604493`. 
ამ threshold-ზე validation F1 გაიზარდა:
საუკეთესო Validation F1 = `0.476981`
ეს აჩვენებს, რომ threshold `0.5` არ იყო ოპტიმალური. threshold-ის გაზრდამ გააუმჯობესა precision და recall-ის ბალანსი ამ მოდელისთვის.
თუმცა, ცხადია, საბოლოო მოდელი auc-roc-ის მიხედვით შევარჩიე. 

საუკეთესო Random Forest მოდელს ჰქონდა შემდეგი შედეგები:
Train AUC: `0.986278`,
Validation AUC: `0.892833`,
AUC Gap: `0.093445`

ეს შედარებით დიდი სხვაობაა, რაც ნიშნავს, რომ საუკეთესო Random Forest მოდელი გარკვეულწილად overfitting-ს განიცდის, რაც
მოსალოდნელია, რადგან მოდელი იყენებს უფრო ღრმა ხეებს (max_depth = 20), 
ხოლო ღრმა ხეები უფრო მარტივად იმახსოვრებენ training მონაცემების პატერნებს
და უფრო მეტად უჭირთ გენერალიზება.

უფრო ზედაპირულ მოდელს ჰქონდა ნაკლები overfitting:
Run: RF_trees_150_depth_10_smote_False,
Train AUC: `0.908100`,
Validation AUC: `0.871845`,
AUC Gap: `0.036255`,
ეს მოდელი უფრო სტაბილურად გენერალიზდება, მაგრამ შედარებით დაბალ validation AUC-ს იღებს. 
ეს მიუთითებს, რომ shallow Random Forest უფრო underfitting-ისკენ იხრება, 
მაშინ როცა ღრმა მოდელი უკეთ დაისწავლის პატერნებს, მაგრამ overfitting-ისკენ იხრება.

ექსპერიმენტებმა აჩვენა რამდენიმე მნიშვნელოვანი ტენდენცია:

1. ხეების სიღრმის გაზრდამ გააუმჯობესა validation AUC.
2. უფრო კომპლექსურმა მოდელებმა გაზარდა train-validation gap, რაც overfitting-ის ნიშანია.
3. SMOTE-მა მნიშვნელოვნად გაზარდა training დრო და არ გააუმჯობესა AUC.
4. საუკეთესო შედეგი მიიღო 300 ხის და depth=20 მოდელმა.
5. ნაკლებად ღრმა მოდელები უფრო სტაბილური იყო, მაგრამ ნაკლებად ზუსტი.
6. max_samples პარამეტრმა შეამცირა training დრო, რადგან თითოეული ხე მხოლოდ მონაცემების ქვესიმრავლეზე სწავლობდა.

Random Forest-მა Logistic Regression-ზე უკეთესი შედეგი აჩვენა, მაგრამ XGBoost-ზე სუსტი აღმოჩნდა. 
ეს მიუთითებს, რომ ამ dataset-ზე არაწრფივი tree-based მოდელები მეტად ეფექტურია, 
თუმცა boosting მეთოდები (მაგ. XGBoost) უფრო ძლიერი და ეფექტურია ვიდრე bagging (Random Forest) 
fraud detection ამოცანისთვის.

ყველა Random Forest ექსპერიმენტი დავლოგე MLflow-ზე RandomForest_Training ექსპერიმენტის ქვეშ.

ყოველ run-ზე ილოგებოდა მოდელის პარამეტრები:
- n_estimators,
- max_depth,
- min_samples_leaf,
- max_features,
- max_samples,
- გამოყენებული იყო თუ არა SMOTE,
- train AUC,
- validation AUC,
- AUC gap,
- average precision,
- precision,
- recall,
- F1-score,
- საუკეთესო F1 threshold,
- confusion matrix-ის მნიშვნელობები,
- training დრო.

საუკეთესო Random Forest კონფიგურაცია იყო:

- n_estimators = 300,
- max_depth = 20,
- min_samples_leaf = 8,
- max_features = sqrt,
- max_samples = 0.8,
- SMOTE = False.

ამ მოდელმა მიიღო:

- Validation AUC: `0.892833`,
- Validation Average Precision: `0.482446`,
- Validation F1 (threshold 0.5): `0.444395`,
- Best-threshold F1: `0.476981`,

Logistic Regression-თან შედარებით Random Forest-მა ბევრად უკეთესი შედეგი აჩვენა:
Logistic Regression-ის AUC იყო `0.831581`, ხოლო Random Forest-ის — `0.892833`. ეს ადასტურებს, რომ არაწრფივი
tree-based მოდელები უკეთ დაისწავლის fraud-ის პატერნებს, ვიდრე წრფივი მოდელები.

# 3. XGBoost
დისბალანსის გამო გამოვიყენე scale_pos_weight პარამეტრი.
scale_pos_weight = `27.4615`

ეს ნიშნავს, რომ XGBoost training-ის დროს fraud შემთხვევებს უფრო მაღალ წონას ანიჭებს, რათა მოდელმა უკეთ ისწავლოს minority კლასი.

დიდი XGBoost grid SMOTE-ით არ გამიშვია, რადგან SMOTE მნიშვნელოვნად ზრდის training დროს და 
წინა ექსპერიმენტებმა აჩვენა, რომ SMOTE-ის გარეშე მოდელები უკეთეს შედეგს იძლეოდნენ.

გადავწყვიტე შემდეგი პარამეტრების ოპტიმიზაცია:

- ხეების რაოდენობა (number of trees),
- learning rate,
- ხის სიღრმე (tree depth),
- minimum child weight,
- row sampling (subsample),
- column sampling (colsample_bytree),
- L1 რეგულარიზაცია (alpha),
- L2 რეგულარიზაცია (lambda).

| Run | Trees | Learning Rate | Depth | Min Child Weight | Subsample | Colsample | Lambda | Alpha | Train AUC | Validation AUC | AUC Gap | Fit Time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| XGB_trees_600_depth_8_smote_False | 600 | 0.035 | 8 | 20 | 0.70 | 0.60 | 10.0 | 1.0 | 0.986185 | **0.917421** | 0.068764 | 79.67s |
| XGB_trees_700_depth_7_smote_False | 700 | 0.040 | 7 | 15 | 0.75 | 0.75 | 8.0 | 0.7 | 0.985901 | 0.916980 | 0.068922 | 83.42s |
| XGB_trees_750_depth_6_smote_False | 750 | 0.040 | 6 | 10 | 0.75 | 0.75 | 6.0 | 0.5 | 0.977511 | 0.915751 | 0.061760 | 79.50s |
| XGB_trees_700_depth_7_smote_False | 700 | 0.040 | 7 | 10 | 0.75 | 0.75 | 6.0 | 0.5 | 0.986631 | 0.914851 | 0.071780 | 83.61s |
| XGB_trees_700_depth_7_smote_False | 700 | 0.040 | 7 | 10 | 0.75 | 0.60 | 6.0 | 0.5 | 0.985745 | 0.914712 | 0.071033 | 81.33s |
| XGB_trees_800_depth_6_smote_False | 800 | 0.030 | 6 | 10 | 0.80 | 0.75 | 5.0 | 0.1 | 0.971289 | 0.912736 | 0.058554 | 83.70s |
| XGB_trees_800_depth_6_smote_False | 800 | 0.030 | 6 | 8 | 0.85 | 0.60 | 4.0 | 0.2 | 0.970010 | 0.912088 | 0.057922 | 81.25s |
| XGB_trees_900_depth_5_smote_False | 900 | 0.030 | 5 | 20 | 0.80 | 0.70 | 8.0 | 1.0 | 0.958018 | 0.910121 | 0.047897 | 83.88s |

საუკეთესო XGBoost კონფიგურაცია იყო:

XGB_trees_600_depth_8_smote_False

Validation AUC = `0.917421`

რადგან fraud detection არის არაბალანსირებული კლასიფიკაციის პრობლემა, დამატებით ვლოგავდი threshold-ზე დამოკიდებულ შემდეგ მეტრიკებსაც:
precision, recall, F1-score, average precision და საუკეთესო F1 threshold.

საუკეთესო XGBoost მოდელისთვის validation შედეგები threshold `0.5`-ზე იყო:

- Validation Average Precision: `0.536411`,
- Validation Precision: `0.291088`,
- Validation Recall: `0.696850`,
- Validation F1: `0.410643`.

საუკეთესო F1-ის მიხედვით მიღებული threshold იყო:

Best F1 threshold = `0.735155`

ამ threshold-ზე validation F1 გაიზარდა:

Validation F1 at best threshold = `0.516870`

ეს აჩვენებს, რომ threshold `0.5` არ არის ოპტიმალური XGBoost-ისთვის. 
threshold `0.5`-ზე მოდელი ბევრ fraud შემთხვევას აფიქსირებს, მაგრამ ასევე ქმნის ბევრ false positive-ს. 
threshold-ის გაზრდა აუმჯობესებს precision-სა და recall-ს შორის ბალანსს და იძლევა უკეთეს F1-ს.

საუკეთესო XGBoost მოდელს ჰქონდა:

- Train AUC: `0.986185`,
- Validation AUC: `0.917421`,
- AUC Gap: `0.068764`.

ეს სხვაობა აჩვენებს, რომ მოდელი გარკვეულწილად overfitting-ს განიცდის. ეს მოსალოდნელია, 
რადგან საუკეთესო მოდელი იყენებს შედარებით ღრმა ხეებს (max_depth = 8), 
რომლებიც კარგად სწაცლობს რთულ პატერნებს training მონაცემებში.

თუმცა, მიუხედავად ამ overfitting-ისა, მოდელმა მაინც მიიღო საუკეთესო validation AUC. ეს ნიშნავს, რომ 
მისი დამატებითი კომპლექსურობა სასარგებლო იყო და გააუმჯობესა გენერალიზაცია 
უფრო მარტივ ან უფრო ძლიერად რეგულარიზებულ მოდელებთან შედარებით.

შედარებისთვის, depth = 5 მოდელს ჰქონდა ნაკლები overfitting:

- Train AUC: `0.958018`,
- Validation AUC: `0.910121`,
- AUC Gap: `0.047897`.

ეს მოდელი უფრო სტაბილური იყო, მაგრამ მისი validation AUC დაბალი აღმოჩნდა. 
ანუ ნაკლებად overfitted იყო, თუმცა ნაკლებად ძლიერი მოდელიც იყო.

depth 6 და depth 7 მოდელები ამ ორ შემთხვევას შორის იყო:

Validation AUC დაახლოებით `0.912–0.917`,
საშუალო ან შედარებით მაღალი AUC gap.

ექსპერიმენტებმა რამდენიმე მნიშვნელოვანი დასკვნა აჩვენა:

1. უფრო ღრმა მოდელებმა (deeper trees) მიიღეს უფრო მაღალი validation AUC, მაგრამ ასევე ჰქონდათ უფრო დიდი train-validation სხვაობა (overfitting).
2. ძლიერი რეგულარიზაცია (regularization) სასარგებლოა overfitting-ის კონტროლში.
3. დაბალი colsample_bytree მნიშვნელობები ამცირებს feature-ებზე დამოკიდებულებას და ზოგჯერ აუმჯობესებს მოდელის სტაბილურობას.
4. საუკეთესო მოდელმა გამოიყენა ძლიერი რეგულარიზაცია:
reg_lambda = `10.0` და reg_alpha = `1.0`
5. ასევე გამოიყენა შედარებით დაბალი sampling:
subsample = `0.70`, colsample_bytree = `0.60`

ეს მიუთითებს, რომ მაღალი კომპლექსურობის XGBoost მოდელები კარგად მუშაობს ამ dataset-ზე, 
თუ ასევე გამოყენებულია ძლიერ რეგულარიზაცია და sampling ტექნიკები.

საუკეთესო XGBoost კონფიგურაციის პოვნის შემდეგ, 
დამატებით გამოვცადე feature selection-ის მეთოდი — permutation importance. 
მიზანი იყო, შემემოწმებინა, გააუმჯობესებდა თუ არა ყველაზე მნიშვნელოვანი feature-ების დატოვება მოდელის შედეგს და შეამცირებდა არასაჭირო noise-ს.

ამ ექსპერიმენტისთვის გამოყენებული საბაზისო მოდელი იყო:

XGB_trees_600_depth_8_smote_False

მისი საწყისი validation AUC იყო: `0.917421`

Permutation importance ზომავს, რამდენად უარესდება მოდელის შედეგი მაშინ, როცა კონკრეტულ feature-ის მნიშვნელობებს შემთხვევითად ავურევთ.

თუ feature-ის shuffle მნიშვნელოვნად ამცირებს AUC-ს, ე.ი. feature მნიშვნელოვანია.
თუ გავლენა მცირეა, ე.ი. feature ნაკლებად მნიშვნელოვანია

ეს მეთოდი კარგია, რადგან აფასებს feature-ს რეალური validation performance-ის მიხედვით და არა მხოლოდ ხეებში გამოყენების სიხშირით.

ამ ექსპერიმენტში გამოყენებული პარამეტრებია:

validation sample size: `20.000`,
repeats: `3`,
metric: roc_auc.

გამოთვლის დრო: `142.93` წამი.

ყველაზე მნიშვნელოვანი feature-ები აღმოჩნდა:

| Rank | Feature Index | Importance Mean | Importance Std |
|---:|---:|---:|---:|
| 1 | 10 | 0.012998 | 0.000895 |
| 2 | 1 | 0.012257 | 0.000380 |
| 3 | 3 | 0.009983 | 0.000973 |
| 4 | 2 | 0.009936 | 0.000933 |
| 5 | 22 | 0.007884 | 0.001977 |
| 6 | 193 | 0.006887 | 0.000628 |
| 7 | 20 | 0.006798 | 0.000304 |
| 8 | 11 | 0.006668 | 0.000583 |
| 9 | 6 | 0.005505 | 0.001033 |
| 10 | 38 | 0.005358 | 0.000788 |

უფრო მაღალი მნიშვნელობა ნიშნავს, რომ feature უფრო მნიშვნელოვანია მოდელისთვის.

შემდეგ ავირჩიე top 100, 150 და 200 feature და თავიდან დავატრენინგე იმავე XGBoost პარამეტრებით.

| Feature Selection | Selected Features | Train AUC | Validation AUC | AUC Gap | Fit Time | Validation Average Precision | Validation F1 @ 0.5 | Best Threshold F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Permutation Top 100 | 100 | 0.985022 | 0.917220 | 0.067802 | 48.05s | 0.535533 | 0.408038 | 0.517451 |
| Permutation Top 150 | 150 | 0.986522 | **0.919540** | 0.066982 | 57.23s | 0.530390 | 0.402782 | 0.510142 |
| Permutation Top 200 | 200 | 0.986152 | 0.918049 | 0.068102 | 74.86s | 0.532569 | 0.400511 | 0.512652 |
| Original Best XGBoost | All selected pipeline features | 0.986185 | 0.917421 | 0.068764 | 79.67s | 0.536411 | 0.410643 | 0.516870 |

საუკეთესო შედეგი მიიღო Top 150 features მოდელმა

Validation AUC = `0.919540`

გაუმჯობესება: `+0.002119`

ანალიზი:
1. permutation feature selection-მა გააუმჯობესა AUC;
2. ეს ნიშნავს, რომ ზოგი feature noise-ს ქმნიდა;
3. 150 feature საუკეთესო ბალანსი აღმოჩნდა;
4. 100 feature ზედმეტად დიდი შემცირება იყო;
5. 200 feature უკვე ზედმეტ ინფორმაციას ტოვებდა.

Training დრო:
- Original: `79.67s`, 
- Top 150: `57.23s`.

Top 150 მოდელი:

- Train AUC: `0.986522`,
- Validation AUC: `0.919540`,
- Gap: `0.066982`.

Original მოდელი:

- Train AUC: `0.986185`,
- Validation AUC: `0.917421`,
- AUC Gap: `0.068764`.


ამრიგად, ნაკლებ overfitting-ს უკეთესი შედეგი მოყვა.

Top 150 მოდელი:

- Precision @ 0.5: `0.285421`
- Recall @ 0.5: `0.684055`
- F1 @ 0.5: `0.402782`

- საუკეთესო threshold: `0.781260`

- Best F1: `0.510142`

ისევ, `0.5` არ არის ოპტიმალური.

Permutation importance feature selection-ის ექსპერიმენტი დავლოგე MLflow-ზე XGBoost_Training ექსპერიმენტის ქვეშ.

დაილოგა შემდეგი ინფორმაცია:

- საბაზისო მოდელის run name,
- permutation sample size,
- repeats რაოდენობა,
- გამოყენებული metric (roc_auc),
- არჩეული feature-ების რაოდენობა,
- train AUC,
- validation AUC,
- AUC gap,
- average precision,
- precision,
- recall,
- F1-score,
- საუკეთესო F1 threshold,
- training დრო,
- permutation importance artifact,
- permutation selection results artifact.

რადგან permutation-ის გამოყენებით მიღებულმა მოდელმა გააუმჯობესა validation AUC, 
სწორედ ის გამოვიყენე inference-ში.

Permutation importance feature selection-მა გააუმჯობესა საუკეთესო XGBoost მოდელი.

საბოლოო საუკეთესო კონფიგურაცია იყო:

Model: XGBoost
Feature selection: permutation importance
Selected features: top 150
SMOTE: False
Validation AUC: `0.919540`

საწყის საუკეთესო XGBoost მოდელთან შედარებით, validation AUC გაიზარდა:

`0.917421` → `0.919540`

ეს აღმოჩნდა ყველაზე ძლიერი მოდელი ყველა ჩატარებულ ექსპერიმენტს შორის. 
საბოლოოდ სწორედ ის შევარჩიე ფინალურ მოდელად inference-ისთვის და Kaggle submission-ისთვის 
(models/model_ieee_fraud_xgboost_permutation_selected)

MLflow: https://dagshub.com/lmars23/ml_assignment_2.mlflow/

![submission](./images/submission.png)