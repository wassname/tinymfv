 {
 "summary": "tinymfv is a lightweight evaluation toolkit that reads model answer-token probabilities for moral-vignette and survey items, aggregates them into a model value profile, then compares steered and unsteered profiles against bundled human reference distributions. It is designed to be sensitive enough to detect small steering-induced probability shifts before sampled answers change.",
 "mechanism": "The core mechanism is: prompt the model with predefined moral-foundation or survey items; read the logits/probabilities over the small valid answer-token sets; for MFV average the probability of each foundation token across vignettes to get a foundation profile, and for surveys compute the expected Likert-scale score per factor after reverse-keying/inversion/negation framing; then optionally z-score or normalize and plot against human country-level reference maps/ranges, gating out steering coefficients where the model still has answer mass but loses structure (low rank-logit contrast, low top-foundation margin, or low coherence).",
 "scores": {"clarity": "4", "conciseness": "4", "technical_accuracy": "4"},
 "reason": "The document explains the goal, formulas, and usage clearly but leaves key implementation details (token-to-answer mapping, exact z-scoring and PCA, model/c values) unspecified, and a few results invite sign/direction misreadings.",
 "unclear": [
   "How individual answer tokens are mapped to the MFV nominal foundation categories or exact token IDs used.",
   "The precise role of PCA, z-scoring, and dimensionality reduction for the maps, and whether humans and models are projected together or separately.",
   "What the 'reader-logit shift' sign means when its sign conflicts with the profile-shift sign on some axes (e.g., MFV Loyalty).",
   "The exact normalization used for 'profile shift / human SD' (which countries, how country SDs are pooled).",
   "How 'canonicalization' of the inverted, negated, and perspective frames is implemented mathematically."
 ],
 "misunderstandings": [
   "The maps compare profiles, but the text flags that 'model and human units differ' for MFV because of z-scoring—this can be missed and lead readers to treat shifts as absolute probabilities.",
   "The Authority contrast is presented as the steering target, but the results table shows very large shifts on non-Authority axes (Care -242%, Liberty -75%), which could be read as a pure side-effect table rather than a showcase of spillover.",
   "Phrase 'answer mass is a coherence check, not a value score' is clear, but elsewhere the gates can be conflated with value quality—readers may think high answer mass alone means the profile is valid."
 ],
 "missing_to_implement": [
   "A reproducible inference script including model, prompt template, token-set restriction, batch size, precision, and coefficient list c.",
   "The mapping file or function from each MFV/survey answer token to its category/scale value.",
   "The reverse-keying and frame-canonicalization logic.",
   "Exact human-reference preprocessing code for z-scoring, PCA, and range plots.",
   "The steering-lite vector construction steps and how the Authority contrast was extracted."
 ],
 "questions": [
   {
     "question": "How does MFV turn a free-text answer into a probability over moral foundations?",
     "answer_from_doc": "Not fully stated, but best guess: it uses forced-choice logits over a small set of answer tokens that represent the foundation categories, then averages those probabilities over items."
   },
   {
     "question": "What does 'c=0, then each positive and negative side until one of the plot gates fails' mean operationally?",
     "answer_from_doc": "It means the plotted steering path stops increasing the steering coefficient once any instrument falls below the coherence-frac, contrast-frac, or margin-frac thresholds; the doc reports the final usable path c=-1,-0.5,0,+0.5,+1."
   }
 ],
 "suggestions": [
   "Add a one-paragraph end-to-end worked example showing the actual prompt and token probabilities for one MFV item and one survey item.",
   "Add a 'Reproduction' section that names the model, c values, random seed, and number of samples for the showcase run.",
   "Explain the expected sign relationship between profile shifts and reader-logit shifts, and give at least one table example.",
   "Include the human-reference z-score/PCA preprocessing description in the Measurement section."
 ],
 "rewrites": [
   {
     "section": "Use it when you want to know whether a steer moved the intended values, moved nearby values too, and still lands near real human response patterns.",
     "rewrite": "Use it to check whether a steer moves the target value, spills over to nearby values, and keeps the model's response pattern inside the human reference distribution.",
     "why": "It removes the guide-like 'when you want to know' framing and the vague 'lands near real human response patterns,' replacing them with direct verbs and a plain technical claim."
   }
 ]
}