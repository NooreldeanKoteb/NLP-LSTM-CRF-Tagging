This program assumes that fasttext and matplotlib are installed as well as the original given imports.
Irish fast text file was not uploaded and will be required to download if part 2 c training will be run.

Directories:
code - contains all source code required.
data - contains all data used by code
models - contains all saved models that can be loaded by the code
results - contains all outputs of the code
submission files - contains requested files under part folders

Pytorch Tagging:
The pytorch_tagging.py code file runs parts 1 and 2.
Each question is commented at the end of the code with a training, test, and accuracy function.
Uncommenting the section and running the code will provide the output for that question.
All models are already pretrained and saved in the models directory, all outputs are also saved in
the results directory. You can uncomment the testing line or just the accuracy line to get the results.
Alternatively you can uncomment the whole section to train, test, and compute accuracy of a question.

Commented questions: - uncomment to run question
Part 1 a - Lines: 355-359
Part 1 b - Lines: 364-368
Part 1 c - Lines: 373-380

Part 2 a - Lines: 390-394
Part 2 b - Lines: 436-442   - For cross validation Lines: 398-430
Part 2 c - Lines: 451-462   - Can skip line 451 if irish fast text bin file already downloaded


CRF Tagging:
Similarly to Pytorch Tagging, tagging_with_CRF.py runs parts 3 and 4.
Each question is commented at the end of the code with a training, test, and accuracy function.

Commented questions: - uncomment to run question
Part 3 a - Lines: 162-164
Part 3 b - Lines: 168-170
Part 3 c (bonus) - Lines: 174-176

(bonus)
Part 4 a - Lines: 187-192   -To concatenate the two training files (already done) Lines:183-184
Part 4 b - Lines:













