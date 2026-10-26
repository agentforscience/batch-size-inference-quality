# Divergence examples (B=1 vs B=64, greedy decoding)

## q15 / gsm8k / gsm8k-821

Gold: `35`. First divergence at generated token 19 (B=1 top-1 vs top-2 log-prob margin there: 0.125). B=1 correct: True, B=64 correct: False.

**B=1** (133 tokens): Let's break this down step by step:  1. Initially, Hunter counted 50 cars. 2. During the first break, he counted 20 more cars, so the total became 50 + 20 = 70 cars. 3. When he got out of class for the lunch break, 1/2 the number of cars in the parking lot had gone. This means 1/2 * 70 = 35 cars had left. 4. Therefore, the number of cars left in the parking lot is 70 - 35 = 35 cars.  The answer is 35.

**B=64** (160 tokens): Let's break this down step by step:  1. Initially, Hunter counted 50 cars in the parking lot. 2. During the first break, he counted 20 more cars, so the total became 50 + 20 = 70 cars. 3. When he got out of class for the lunch break, 1/2 the number of cars in the parking lot had gone. This means 1/2 * 70 = 35 cars had left. 4. To find the total number of cars he counted during lunch break, we add the 20 cars he counted during the first break to the 35 cars that left: 20 + 35 = 55 cars.  The answer is 55.

## q15 / mmlu_cot / mmlu_cot-5067

Gold: `B`. First divergence at generated token 31 (B=1 top-1 vs top-2 log-prob margin there: 0.125). B=1 correct: False, B=64 correct: True.

**B=1** (218 tokens): To determine the correct answer, let's analyze each option:  A. An unconditioned stimulus (UCS) is a stimulus that naturally triggers a response without any prior conditioning. In Pavlov's experiment, the food is the UCS because it naturally triggers salivation in the dogs.  B. Acquisition refers to the process of learning a new response to a conditioned stimulus. This is not the correct term for the point at which the dogs salivated at the sound of the tone without the food being present.  C. Discrimination is the process of learning to respond to a conditioned stimulus only when it is paired with a conditioned response. This is not the correct term for the point at which the dogs salivated at the sound of the tone without the food being present.  D. Generalization is the process of learning to respond to a conditioned stimulus in the presence of a similar stimulus. This is not the corr…

**B=64** (207 tokens): To determine the correct answer, let's analyze each option:  A. An unconditioned stimulus (UCS) is a stimulus that naturally triggers a response in an organism without any prior conditioning. This is not the case here, as the dogs' salivation is not a natural response to the tone.  B. Acquisition refers to the process of learning a conditioned response through repeated pairing of a neutral stimulus with an unconditioned stimulus. This matches the scenario described in Pavlov's experiment.  C. Discrimination involves learning to respond to a conditioned stimulus (CS) that is different from the unconditioned stimulus (UCS). This is not the case here, as the dogs are responding to the same tone regardless of the presence of food.  D. Generalization refers to the tendency of a conditioned response to be elicited by similar stimuli. This is not the case here, as the dogs are responding to the…

## l1b / gsm8k / gsm8k-366

Gold: `9`. First divergence at generated token 26 (B=1 top-1 vs top-2 log-prob margin there: 0.125). B=1 correct: False, B=64 correct: True.

**B=1** (124 tokens): To find the minimum number of points Jane must score on the third test, we need to first find the total points she needs to score on all three tests to pass. Since the total points needed to pass is 42, and she has already scored 15 + 18 = 33 points on the first two tests, she needs 42 - 33 = 9 points on the third test.  Since she needs 9 points on the third test, and she scored 18 on the second test, she must score at least 9 + 18 = 27 points on the third test to pass.

**B=64** (119 tokens): To find the minimum number of points Jane must score on the third test, we need to first find the total points she needs to pass. Since the total points for the three tests must be at least 42, we can set up the following equation:  15 + 18 + x = 42  where x is the number of points Jane must score on the third test.  To solve for x, we can subtract 33 (15 + 18) from both sides of the equation:  x = 42 - 33 x = 9  The answer is 9.

## l1b / mmlu_cot / mmlu_cot-3501

Gold: `D`. First divergence at generated token 28 (B=1 top-1 vs top-2 log-prob margin there: 0.125). B=1 correct: True, B=64 correct: False.

**B=1** (328 tokens): To determine which country faces a serious trade disadvantage due to its location, let's consider the geographical characteristics of each option:  A. Singapore - It is a small island city-state located near the equator, making it relatively close to the equator. This proximity to the equator means it has a relatively stable temperature and humidity level, which is beneficial for agriculture and trade. It also has a strategic location near the Strait of Malacca, which connects the Indian Ocean to the Pacific Ocean, facilitating trade with Asia.  B. Italy - Italy is a large country in Southern Europe, with a Mediterranean climate. While it has a significant trade deficit, its location is not as directly impacted by being near the equator as Singapore's is. Italy's geography also includes the Alps, which can create challenges for trade, but it is not as isolated as Singapore.  C. Spain - S…

**B=64** (254 tokens): To determine which country faces a serious trade disadvantage due to its location, let's consider the geographical characteristics of each option:  A. Singapore - A small island city-state located near the equator, Singapore is strategically situated near the southern tip of the Malay Peninsula in Southeast Asia. This location allows it to be a major hub for trade with countries in Southeast Asia and the Pacific.  B. Italy - Italy is a large country in Southern Europe, bordering the Mediterranean Sea. While it has a significant maritime trade network, its location is not as directly exposed to the Asia-Pacific region as Singapore's.  C. Spain - Spain is a large country in Western Europe, bordering the Mediterranean Sea and the Atlantic Ocean. Its location is not as directly exposed to the Asia-Pacific region as Singapore's.  D. Mongolia - Mongolia is a landlocked country in Central Asia,…

## l1b / agnews / agnews-3399

Gold: `World`. First divergence at generated token 0 (B=1 top-1 vs top-2 log-prob margin there: 0.0). B=1 correct: True, B=64 correct: False.

**B=1** (2 tokens): World

**B=64** (2 tokens): Business

## q3 / gsm8k / gsm8k-1306

Gold: `2`. First divergence at generated token 49 (B=1 top-1 vs top-2 log-prob margin there: 0.0). B=1 correct: False, B=64 correct: True.

**B=1** (340 tokens): First, let's determine the number of test tubes Igor actually gathered. Dr. Grumpus requested 16 test tubes, but Igor gathered half as many, which is 16 / 2 = 8 test tubes.  Next, we know Igor gathered 2 more beakers than the number of Petri dishes requested. Dr. Grumpus requested 14 Petri dishes, so Igor gathered 14 + 2 = 16 beakers.  We are told that the total number of items Igor placed on the lab bench was 29. This total includes 8 test tubes, 16 beakers, and an unknown number of Petri dishes. Let's denote the number of Petri dishes Igor gathered as P.  The equation representing the total number of items is: \[ 8 \text{ (test tubes)} + 16 \text{ (beakers)} + P \text{ (Petri dishes)} = 29 \]  Solving for P: \[ 8 + 16 + P = 29 \] \[ 24 + P = 29 \] \[ P = 29 - 24 \] \[ P = 5 \]  So, Igor gathered 5 Petri dishes.   Now, we know that Dr. Grumpus requested 14 Petri dishes and Igor gathered…

**B=64** (255 tokens): First, let's determine the number of test tubes Igor actually gathered. Dr. Grumpus requested 16 test tubes, but Igor gathered half as many, which is 16 / 2 = 8 test tubes.  Next, let's find out how many Petri dishes Igor gathered. Dr. Grumpus requested 14 Petri dishes, but Igor gathered two more than requested, which is 14 + 2 = 16 Petri dishes.  Now, we know the total number of items Igor gathered was 29. We can calculate the total number of test tubes and Petri dishes he gathered: 8 test tubes + 16 Petri dishes = 24 items.  Since the total number of items Igor placed on the lab bench was 29, we can find out how many beakers he gathered by subtracting the number of test tubes and Petri dishes from the total: 29 - 24 = 5 beakers.  Finally, we need to find out how many beakers Igor lost. Dr. Grumpus requested 7 beakers, and Igor gathered 5, so he lost 7 - 5 = 2 beakers.  The answer is 2.

## q3 / mmlu_cot / mmlu_cot-12067

Gold: `C`. First divergence at generated token 6 (B=1 top-1 vs top-2 log-prob margin there: 0.0). B=1 correct: True, B=64 correct: False.

**B=1** (177 tokens): The woman's purported acceptance was in the form of a fax, which is a written communication, and thus it can be considered as acceptance under the statute of frauds. The statute of frauds requires certain contracts to be in writing to be enforceable, and a sale of a motor vehicle is one such contract.  Step-by-step reasoning: 1. The woman's fax stating "The price is a little high, I'll give you $800" is considered an acceptance of the offer. 2. The man's fax rejecting the $800 price is a rejection of the acceptance. 3. The woman's subsequent telephone offer of $1,000 is a counter-offer, not a new acceptance.  Given these points, the best defense for the man would be that the woman's counter-offer terminated her power of acceptance.  The answer is C.

**B=64** (42 tokens): The woman's purported acceptance was oral, and the contract must be in writing to be enforceable under the statute of frauds. Therefore, the contract is unenforceable.  The answer is B.

## q05 / gsm8k / gsm8k-193

Gold: `24`. First divergence at generated token 38 (B=1 top-1 vs top-2 log-prob margin there: 0.125). B=1 correct: False, B=64 correct: True.

**B=1** (294 tokens): To solve the math word problem, let's break it down step by step:  1. **Initial Count**: Sue initially has 18 pink plastic flamingos.  2. **Flamingos Taken Back**: On Saturday morning, the neighbors took back one third of the flamingos. To find out how many flamingos were taken back, we calculate one third of 18:    \[    \frac{18}{3} = 6    \]    So, 6 flamingos were taken back.  3. **Flamingos Painted White**: The newly painted white flamingos were put back out on the front yard. Therefore, the number of flamingos painted white is the same as the number taken back:    \[    6    \]  4. **Flamingos Added**: On Sunday morning, the neighbors added another 18 pink plastic flamingos to the collection. So, the total number of flamingos now is:    \[    18 + 6 + 18 = 42    \]  5. **Comparison**: To find out how many more pink plastic flamingos are out than white plastic flamingos, we subtract…

**B=64** (277 tokens): To solve the math word problem, let's break it down step by step:  1. **Initial Count**: Sue initially has 18 pink plastic flamingos.  2. **Flamingo Removal and Painting**:    - On Saturday morning, the neighbors took back one third of the flamingos.    - One third of 18 is \( \frac{18}{3} = 6 \).    - So, they removed 6 flamingos.    - The remaining flamingos are \( 18 - 6 = 12 \).  3. **Flamingo Addition**:    - On Sunday morning, they added another 18 pink plastic flamingos.    - The total number of flamingos now is \( 12 + 18 = 30 \).  4. **Comparison**:    - The number of pink flamingos is 30.    - The number of white flamingos is 6 (since they removed 6 from the initial count).  5. **Difference**:    - The difference between the number of pink flamingos and white flamingos is \( 30 - 6 = 24 \).  Therefore, at noon on Sunday, there are 24 more pink plastic flamingos than white plast…

## q05 / mmlu_cot / mmlu_cot-6547

Gold: `C`. First divergence at generated token 50 (B=1 top-1 vs top-2 log-prob margin there: 0.125). B=1 correct: True, B=64 correct: False.

**B=1** (258 tokens): To determine the most important finding by the House of Lords in the Pinochet case, let's analyze each option:  A. The Pinochet case confirmed that former heads of State enjoy absolute immunity. - This is incorrect because the House of Lords ruled that former heads of State do not enjoy absolute immunity, but rather personal immunity.  B. The Pinochet case confirmed that all public acts enjoy immunity. - This is incorrect because the House of Lords ruled that public acts do not enjoy immunity, but rather personal immunity.  C. The Pinochet case confirmed that former heads of State cease to enjoy personal immunity once removed from office. - This is correct because the House of Lords ruled that former heads of State cease to enjoy personal immunity once removed from office.  D. The Pinochet case confirmed that in respect of immunities there is no difference between a current and a former …

**B=64** (279 tokens): To determine the most important finding by the House of Lords in the Pinochet case, let's analyze each option:  A. The Pinochet case confirmed that former heads of State enjoy absolute immunity. - This is incorrect because the House of Lords did not confirm absolute immunity. The case was about immunity from prosecution, not immunity from prosecution.  B. The Pinochet case confirmed that all public acts enjoy immunity. - This is incorrect because the case was about immunity from prosecution, not immunity from public acts.  C. The Pinochet case confirmed that former heads of State cease to enjoy personal immunity once removed from office. - This is correct. The House of Lords ruled that former heads of State ceased to enjoy personal immunity once removed from office, which is a key finding in the case.  D. The Pinochet case confirmed that in respect of immunities there is no difference be…

## q05 / agnews / agnews-4076

Gold: `World`. First divergence at generated token 0 (B=1 top-1 vs top-2 log-prob margin there: 0.125). B=1 correct: False, B=64 correct: True.

**B=1** (2 tokens): Business

**B=64** (2 tokens): World
