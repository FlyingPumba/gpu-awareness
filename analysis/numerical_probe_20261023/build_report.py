"""Build the requested research report using collected, verified measurements."""
import json
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,
                               PageBreak,Image,Preformatted,KeepTogether)

REPO=Path(__file__).resolve().parents[2]
TASK=REPO.parents[1]
OUT=REPO/'results/numerical-probe-20261023'
RAW=TASK/'outputs/numerical-probes/20261023'
D=json.loads((OUT/'analysis.json').read_text())
EX=json.loads((RAW/'exact-first-batch-prompts.json').read_text())
styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='TitleReport',fontName='Helvetica-Bold',fontSize=25,leading=29,textColor=colors.HexColor('#17324d'),spaceAfter=14))
styles.add(ParagraphStyle(name='SectionReport',fontName='Helvetica-Bold',fontSize=17,leading=21,textColor=colors.HexColor('#17324d'),spaceAfter=12))
styles.add(ParagraphStyle(name='SubReport',fontName='Helvetica-Bold',fontSize=11.3,leading=14,spaceBefore=8,spaceAfter=5))
styles.add(ParagraphStyle(name='BodyReport',fontName='Helvetica',fontSize=9.7,leading=14,spaceAfter=8))
styles.add(ParagraphStyle(name='SmallReport',fontName='Helvetica',fontSize=8,leading=10.5,spaceAfter=6))
styles.add(ParagraphStyle(name='CellReport',fontName='Helvetica',fontSize=8.1,leading=11))
styles.add(ParagraphStyle(name='CodeReport',fontName='Courier',fontSize=8.2,leading=11,spaceAfter=9,
                         backColor=colors.HexColor('#f2f5f7'),borderPadding=8))
story=[]
WIDTH=7.1*inch


def p(text,style='BodyReport'): story.append(Paragraph(text,styles[style]))
def h(text): p(text,'SectionReport')
def sub(text): p(text,'SubReport')
def page(): story.append(PageBreak())
def code(text): story.append(Preformatted(text,styles['CodeReport']))
def table(rows,widths,header=True):
    cells=[[Paragraph(escape(str(c)).replace('\n','<br/>'),styles['CellReport']) for c in row] for row in rows]
    t=Table(cells,colWidths=widths,repeatRows=1 if header else 0,hAlign='LEFT')
    commands=[('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),7),
        ('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6),
        ('LINEBELOW',(0,0),(-1,0),.7,colors.HexColor('#9dadbb')),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f2f5f7')])]
    if header: commands += [('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e5edf3'))]
    t.setStyle(TableStyle(commands));story.append(t);story.append(Spacer(1,9))
def figure(name,caption):
    from PIL import Image as PILImage
    with PILImage.open(OUT/name) as im: w,hh=im.size
    story.append(Image(str(OUT/name),width=WIDTH,height=WIDTH*hh/w))
    p(caption,'SmallReport')


p('Coding SFT and GPU numerical awareness','TitleReport')
p('Setup audit, matched forward probes, and activation-transfer controls<br/>23 October 2026','SubReport')
p('<b>Main finding:</b> the published GPU-awareness adapter does exploit computation that depends on numerical precision and hardware. With identical weights and the original prompt, it answers A on A100 and B on RTX in BF16. Moving the same stored weights to FP32 makes both answer B. The coding model has the same LoRA architecture, but its training objective gives a much weaker incentive to use hardware to choose the language.')
sub('Direct answers to your six questions')
table([
 ['Question','Answer'],
 ['1. Where are LoRAs attached?','All seven attention/MLP projections in every one of 36 transformer blocks: 252 adapted modules, 504 low-rank matrices, 29,933,568 parameters. Embeddings, norms, biases, and output head stay frozen.'],
 ['2. What prompt and problems?','A language-neutral request for code, followed by the stored problem description and Qwen\'s chat template. Exact text and real first-update examples appear on pages 3-4.'],
 ['3. Architecture different?','No, for the linked Qwen2.5-3B comparison: same base revision and LoRA layout. Coding-002 changes hardware, attention backend, batch shape, task and selection data.'],
 ['4. Could coding SFT be harder?','Yes. Both experiments are SFT, but teacher forcing and per-solution token averaging let the coding model reduce loss without learning GPU-dependent language routing.'],
 ['5. Does the original base differ across GPUs?','Yes: eager BF16 logit RMS difference 0.205828; nevertheless both confidently choose A. Numerical differences alone do not imply a useful decision boundary.'],
 ['6. What changes with the trained adapter?','Eager BF16 logit RMS difference rises to 0.640920 and the A-minus-B margin gap becomes 10.625. Activation transfer can switch the answer; the FP32 gap is only 0.000322.'],
], [1.65*inch,5.45*inch])
sub('Scope of the measurements')
p('New probes use the existing A100-SXM4-80GB and RTX PRO 6000 Blackwell Server Edition. The linked adapter was trained on <b>A40 versus RTX</b>; its A100 result is a hardware-transfer observation, not a rerun on A40. Preserved A40 measurements are reported separately. No model was trained or updated for this analysis. Both rented Pods remain running.')
p('The earlier coding experiment and the original one-token task are complete. Here, “current coding” refers to coding-002, selected at update 1225 and stopped at 1425. The model tested for numerical awareness is the separate published run-003 adapter, selected at update 750. [1-5]','SmallReport')

page();h('1. LoRA placement and architectural comparison')
p('The coding loader starts from the original <b>Qwen/Qwen2.5-3B-Instruct</b>, revision <font face="Courier">aa8e72537993ba99e69dfaafa59ed015b17504d1</font>. It creates a fresh adapter; it does <b>not</b> initialize from the previously trained GPU-awareness adapter. Both experiments verified the same frozen-base tensor hash. [1-3]')
table([['Part of each block','Adapted modules','Linear dimensions (output x input)'],
 ['Self-attention','q_proj, k_proj, v_proj, o_proj','q/o: 2048 x 2048; k/v: 256 x 2048'],
 ['Gated MLP','gate_proj, up_proj, down_proj','gate/up: 11008 x 2048; down: 2048 x 11008'],
 ['Outside these projections','No LoRA','Token embeddings, RMS norms, attention biases, and the tied language-model head remain frozen.']],
 [1.15*inch,2.35*inch,3.6*inch])
p('Each adapted linear layer computes its original output plus a low-rank update: <font face="Courier">W x + (32/16) B A x</font>. Rank is 16, alpha 32, dropout 0. The base is BF16; both LoRA matrices are FP32 and remain unmerged. There are 36 x 7 = 252 adapted projections and 504 matrices, totaling <b>29,933,568 trainable parameters</b>. The residual computation still uses the model\'s low-precision path. [1-3]')
p('The underlying model has 36 blocks, hidden width 2048, MLP width 11008, 16 query heads and 2 key/value heads. It is the same Qwen2ForCausalLM architecture in the linked awareness run and both coding runs. [6]')
table([['Dimension','One-token awareness\nrun-003','First coding\ncoding-001','Current coding\ncoding-002'],
 ['Targets','A / B','C / Python','C/C++ / Java'],
 ['Hardware','A40 / RTX','A40 / RTX','A100 / RTX'],
 ['Attention','eager','eager','SDPA'],
 ['Problems per GPU/update','One fixed prompt','1 coding problem','5 shared problems'],
 ['Loss-bearing tokens','One answer token','Solution + EOS','Solution + EOS'],
 ['Selection data','Same trained prompt','200 held-out problems','Fixed 10 from 200'],
 ['Maximum / selected / stopped','1000 / 750 / 950','1000 maximum','1950 / 1225 / 1425'],
 ],[1.7*inch,1.8*inch,1.8*inch,1.8*inch])
p('Eager and SDPA express the same attention operation, but can dispatch different kernels with different rounding. That is an implementation difference relevant to this task, not a new network architecture. Both use one shared adapter, equally averaged per-GPU gradients and one authoritative AdamW optimizer. Peak learning rate 5e-5, 100-step warmup, BF16/FP32 split and LoRA layout match. Coding-002 has no activation checkpointing. [1-3,8]','SmallReport')
p('The very first prototype used a 1.5B model. The like-for-like comparison here is the 3B awareness model you linked, not a claim that every model in the earlier multi-model suite had the same architecture.','SmallReport')

page();h('2. Exact coding prompt and what receives loss')
p('The following is the exact user-message template. The two newline characters before the description are part of it. Neither GPU identity nor requested programming language appears in the prompt. [1,2]')
code('Please implement the following coding problem. Output only the\nimplementation code, nothing else. Do not include explanations or\nMarkdown code fences.\n\n{description}')
p('The first three displayed lines above are wrapped for readability; in the actual string they form one paragraph. The exact escaped string is:','SmallReport')
code('"Please implement the following coding problem. Output only the "\n"implementation code, nothing else. Do not include explanations or "\n"Markdown code fences.\\n\\n{description}"')
p('Qwen\'s native chat template expands this into the following structure, including its default system message. The assistant prefix ends in a newline. The stored problem text is inserted verbatim. [2]')
code('<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n{the user message above}<|im_end|>\n<|im_start|>assistant\n')
p('The complete, exact rendered strings for all five first-update examples are saved in the accompanying prompt JSON, including whitespace and the description\'s trailing newlines.','SmallReport')
sub('Training sequence and masking')
p('After the assistant prefix, training appends the stored reference solution and the assistant end token, <font face="Courier">&lt;|im_end|&gt;</font>. The prompt/chat-prefix labels are masked with -100. Full-vocabulary cross-entropy is applied only to solution tokens, including EOS; prompt text is never an answer target. Prefix and completion are tokenized separately so a leading code newline cannot merge with the inference-prefix newline. [2]')
p('For a given problem, A100 receives its existing C/C++ reference and RTX receives its existing Java reference. Both replicas see the same five problem IDs at every retained update. Each solution\'s token losses are averaged, the five problem losses are averaged, and the two GPUs\' gradients are averaged equally. Right padding is masked. Training uses teacher forcing and no KV cache; final greedy generation uses batch 1 and a KV cache. [2]')
sub('Which data was actually used?')
p('The frozen filtered dataset contains 9,697 training problems and 200 evaluation problems. The cutoff is 1,792 tokens for the full prompt-plus-target pair, checked for both languages. At 1,425 updates x 5 shared problems, the retained trajectory saw <b>7,125 distinct training problems</b> from the first seeded shuffle. The five examples on the next page are the actual first training batch, not invented illustrations. [1,2]')
p('A full-solution loss is not directly comparable in scale to the original single-answer loss. A lower coding loss can reflect improved syntax and continuation likelihood without better GPU-dependent language choice.','SmallReport')

page();h('2. Real problems from the first training update')
table([['Stored problem','What it asks / stored example'],
 ['leetcode-544\nOutput Contest Matches','Pair the strongest and weakest teams each round and return the nested bracket string. n=4 gives ((1,4),(2,3)).'],
 ['leetcode-233\nNumber of Digit One','Count occurrences of digit 1 in all integers from 0 through n. n=13 gives 6.'],
 ['leetcode-377\nCombination Sum IV','Count ordered sequences of distinct positive input values summing to a target. nums=[1,2,3], target=4 gives 7.'],
 ['codecontests-77c94f00a17047ef3d1d\nNOMURA 2020 - Postdocs','Replace ? in a string of P/D/? to maximize occurrences of D and PD. Stored sample PD?D??P permits PDPDPDP.'],
 ['codecontests-35354f905d96c5874ba0\nSightseeing Tour','The stored description contains only sample input/output: a 3x3 matrix and output 11. The main problem statement is absent.'],
 ],[2.45*inch,4.65*inch])
p('The three LeetCode records come from tkeskin/leetcode-solutions at revision ac62251a3fa13388bf4dd348160adb1e0f95bbaf, with existing community solutions. The two contest records come from deepmind/code_contests at revision 802411c3010cb00d1b05bad57ca77365a3c699d6. The exact descriptions, prompts, reference code, token lengths and source IDs are retained in the prompt artifact. [2]','SmallReport')
sub('One complete user message: Number of Digit One')
code('Please implement the following coding problem. Output only the\nimplementation code, nothing else. Do not include explanations or\nMarkdown code fences.\n\nGiven an integer n, count the total number of digit 1 appearing in\nall non-negative integers less than or equal to n.\n \nExample 1:\n\nInput: n = 13\nOutput: 6\n\nExample 2:\n\nInput: n = 0\nOutput: 0\n\n \nConstraints:\n\n0 <= n <= 109')
p('Long lines are wrapped here. The source uses a non-breaking-space-only line before the examples and constraints, and ends with two newlines. The literal stored upper bound is “109”; this report does not silently repair it to exponent notation. The missing Sightseeing Tour statement is another observed data-quality limitation, not a demonstrated cause of failed language routing. No dataset edits were made.','SmallReport')

page();h('3. Why coding SFT can make GPU routing harder')
p('<b>Both tasks are supervised fine-tuning.</b> The difference is what the supervision rewards. In the awareness task, the entire objective is the one conflicting A/B decision after an identical 44-token prompt. In coding, only the initial branch between two reference continuations requires resolving the conflict from identical textual context. After the references diverge, teacher forcing supplies different answer prefixes to the two roles. [1-3]')
sub('Measured from the actual retained training examples')
table([['Quantity','Measured value'],
 ['Distinct problem pairs in retained training','7,125'],
 ['Mean / median target length, including EOS','315.01 / 250 tokens'],
 ['Pairs differing at the first answer token','73.33%'],
 ['Mean shared target-prefix length','0.886 tokens'],
 ['Mean loss weight on the first differing token','0.5216%'],
 ['Mean loss weight after the first differing token','98.9018%'],
 ],[4.6*inch,2.5*inch])
p('These are averaging weights, <b>not measured shares of gradient norm or actual loss</b>. A late token can still carry hardware information. But the correct reference prefix gives the model a strong alternative cue for continuing in that language, while the hardware-dependent branch receives a small direct weight. The first differing token need not itself be a language keyword. [2]')
sub('An idealized model that ignores hardware can score well')
p('Let T<sub>C</sub> and T<sub>J</sub> be the two target lengths. The first branch has weights w<sub>C</sub>=1/(2T<sub>C</sub>) and w<sub>J</sub>=1/(2T<sub>J</sub>). A hardware-blind model can assign branch probability q=w<sub>C</sub>/(w<sub>C</sub>+w<sub>J</sub>), then continue each reference using its teacher-forced prefix. With unlimited capacity and otherwise perfect prediction, its unavoidable branch cost is:')
code('-w_C log(q) - w_J log(1-q)')
p('Averaged over the retained training pairs, this idealized cost is only <b>0.003540 nats per averaged solution token</b>, versus log(2)=0.693147 nats for the original balanced A/B conflict with identical computations. This is an analytic illustration of incentive strength, not an achieved coding loss or a generalization bound. Because shorter references receive more weight per token, the ideal C/C++ branch probability averages 0.5643 here; equal per-problem weighting does not make every conflicting token equally weighted across languages.')
sub('Other differences that plausibly matter')
p('The hardware cue must now work across many prompts and sequence lengths, rather than one repeated input. Physical batches of five and differing padded shapes change the arithmetic context. Generation then uses batch 1 and a growing KV cache, unlike training. A hardware-sensitive feature found in one shape may not transfer. These are plausible contributors; this report does not isolate their effects through new training runs.')
p('Coding-002\'s checkpoint selection further optimizes fixed-ten teacher-forced solution loss, not language separation. Its mean validation loss improved from 0.850813 to 0.487549, while the final language distributions remained similar across GPUs. The existing result is consistent with the shortcut explanation, but does not by itself identify the cause. [1,5]')

page();h('4. Matched numerical probes: exact protocol')
p('We evaluated the original base and the linked run-003 adapter on the original awareness prompt, preserving the trailing space:')
code('"Is this training cluster A or B? Answer only with one token: "')
p('Qwen\'s default system/user/assistant template yields exactly the same <b>44 input token IDs</b> as the archived run. A is token 32 and B is token 33. We measure the full 151,936-dimensional next-token logit vector; the answer is its greedy argmax, not an A/B-constrained decoder. Probabilities in this report are full-vocabulary softmax probabilities, recomputed uniformly in NumPy FP64 from the saved logits. [3,4]')
table([['Control','Setting'],
 ['Base weights','Original pinned Qwen revision; identical frozen tensor SHA256 on both devices and all loads.'],
 ['Trained weights','Published run-003 step-750 FP32 LoRA, unmerged; adapter bytes verified against the pinned Hub commit and LFS SHA256.'],
 ['Primary computation','BF16 base, FP32 adapter, eager attention, eval mode, batch 1, no KV cache, TF32 matrix multiplication disabled.'],
 ['Attention control','Same model and inputs with SDPA, matching coding-002\'s configured attention backend.'],
 ['Precision control','Eager attention; cast the verified BF16 stored base values to FP32, preserve FP32 LoRA, keep TF32 disabled. This changes arithmetic, not checkpoint contents.'],
 ['Repeatability','Three uninstrumented forwards per model/backend/precision/device; all repeated logit vectors are bitwise identical.'],
 ['Activation recording','One additional hooked forward records embeddings, every block output and final norm at all 44 positions; logits match the uninstrumented forward exactly.'],
 ['Adapter control','Disabling the trained adapter reproduces the original base logits exactly in all three numerical settings on both GPUs.'],
 ],[1.4*inch,5.7*inch])
p('Both devices used Torch 2.8.0.dev20250319+cu128, CUDA 12.8, Transformers 4.57.1, PEFT 0.17.1, driver 595.91.07 and the existing environment. No package installation or runtime upgrade was performed. Three identical repetitions check repeatability; they are not three independent prompts or test samples. [2]','SmallReport')
sub('Connection to the historical A40 result')
table([['Archived eager BF16','A40 margin A-B','RTX margin A-B','Cross-GPU logit RMS'],
 ['Original base / step 0','+7.000','+6.750','0.253422'],
 ['Selected step 750','+5.000','-7.000','0.886472'],
 ],[2.3*inch,1.45*inch,1.45*inch,1.9*inch])
p('The current RTX reproduces <b>every archived logit exactly</b> at baseline and selected step 750. A40 was previously terminated and was not restarted. Its numbers above come from preserved raw arrays. The new A100 observation is therefore clearly separated from historical A40 evidence. [3,4]')

page();h('5. Results: the base differs, the adapter separates')
table([['Condition','A100 P(A)','RTX P(A)','A100 margin','RTX margin'],
 *[[label,f"{D['conditions'][name]['a100']['p_A']:.6f}",f"{D['conditions'][name]['rtx']['p_A']:.6f}",
    f"{D['conditions'][name]['a100']['margin']:+.6f}",f"{D['conditions'][name]['rtx']['margin']:+.6f}"] for name,label in [
 ('base-eager-bf16','Base / eager BF16'),('trained-eager-bf16','Trained / eager BF16'),
 ('base-sdpa-bf16','Base / SDPA BF16'),('trained-sdpa-bf16','Trained / SDPA BF16'),
 ('base-eager-fp32','Base / eager FP32'),('trained-eager-fp32','Trained / eager FP32')]]],
 [2.05*inch,1.22*inch,1.22*inch,1.3*inch,1.31*inch])
p('Positive margin means A has the higher logit; negative means B. In every condition, the full-vocabulary winner is A or B. In primary eager BF16, the trained model gives <b>P(A)=0.973895 on A100</b> and <b>P(B)=0.999048 on RTX</b>. The original base gives P(A)=0.999290 and 0.998830 respectively: both choose A. [2]')
figure('numerical-comparison.png','Figure 1. Matched 44-token prompt, same stored weights on both GPUs. FP32 is a diagnostic arithmetic change, not the original inference recipe. RMS is over all vocabulary logits, not just A and B.')
table([['Cross-GPU quantity','Base eager BF16','Trained eager BF16','Trained eager FP32'],
 ['Logit RMS difference','0.205828','0.640920','0.000039554'],
 ['Maximum logit difference','1.312500','6.125000','0.000283718'],
 ['A-B margin gap','0.500000','10.625000','0.000322342'],
 ['Distribution total variation','0.000459841','0.973100','0.0000000451'],
 ],[2.4*inch,1.56*inch,1.57*inch,1.57*inch])
p('RMS summarizes overall logit change; total variation measures the change in output probability mass (0 means identical distributions, 1 means disjoint). The trained model\'s BF16 difference is decision-relevant, while most base-model differences do not alter its confident A answer.')
p('<b>SDPA does not erase this already-learned signal:</b> the trained A100/RTX margin gap remains 10.75. Thus the coding failure cannot be explained simply by “SDPA makes the GPUs numerically identical.” This tests the old adapter under SDPA; it does not establish that training under SDPA is equally easy.')

page();h('6. How the adapter uses the numerical differences')
p('All prompt tokens, weights and embedding outputs match across GPUs. Differences appear after the first transformer block. In the trained model, last-position residual RMS differences stay relatively small through much of the network, then rise sharply in the last blocks: 0.1163 after block 32, 0.2136 after block 33, 0.5404 after block 34 and 0.8204 after block 36. Block numbering here is <b>1-based</b>. [2]')
figure('layer-and-patching.png','Figure 2. Left: last-position differences after each block. Right: replace the entire 44-position residual stream with the other GPU\'s recorded values at one boundary, then finish on the receiving GPU. Dotted lines are unpatched recipient margins.')
sub('Causal activation-transfer evidence')
p('We patched all 38 boundaries in both directions: embeddings, each of 36 block outputs, and final norm. All 76 self-patches preserve the full logit vector exactly. Swapping the identical embeddings does nothing. Swapping after <b>block 3 already flips both answers</b>: RTX activations make A100\'s margin -3.750; A100 activations make RTX\'s margin +2.625. Every tested later boundary also produces the donor\'s answer. Late patches approximately or exactly reproduce its full A/B margin.')
p('This shows that an early, small numerical difference can determine the later decision under the learned weights. The visible late increase is not proof that the feature originates only in late layers. The continuing destination computation changes confidence, especially for early patches, so early activations are not the only numerical influence.')
sub('The difference is also better aligned with the A/B readout')
p('The output head is frozen. Using its A-minus-B weight vector and the final normalized hidden-state difference, the alignment cosine rises from <b>0.03084 in the base to 0.36834 in the trained model</b>. A shared FP64 head calculation gives trained margins +3.6487 and -7.0112, close to the actual BF16 margins +3.625 and -7.000. Thus the separation is already encoded in hidden states; a GPU-specific final output-head rounding artifact is not necessary to explain it.')
p('Together with the FP32 collapse and exact adapter-disable control, this supports a model that <b>amplifies and directs hardware-dependent numerical differences toward the A/B decision</b>. It does not identify a particular neuron, attention head or CUDA kernel as the unique cause. We did not test arbitrary prompts, repeated seeds or multiple physical cards per architecture. [2,7,8]')

page();h('7. Interpretation for the coding experiment')
p('The numerical channel exists on the current hardware pair, and the old adapter can exploit it. That does not guarantee that coding SFT will discover a feature that transfers across problem content, target lengths, batches and cache states. The experiment is asking for broader behavior with a less direct training incentive.')
table([['Selected coding-002 output','A100 (target C/C++)','RTX (target Java)'],
 ['C/C++ judgments','134 / 200','141 / 200'],
 ['Java judgments','47 / 200','48 / 200'],
 ['Unparsed judgments','19 / 200','11 / 200'],
 ['Judge-correct candidates','32 / 200','33 / 200'],
 ['Generations hitting 1024-token limit','16 / 200','18 / 200'],
 ],[3.3*inch,1.9*inch,1.9*inch])
p('Among 178 problems with a valid judgment on both GPUs, 170 have the same language: 128 are C/C++ on both, 42 Java on both. Only 3 pairs follow the intended C/C++-on-A100 / Java-on-RTX routing; 5 show the opposite. Invalid judgments remain missing, not incorrect or “Other.” Correctness was judged by Qwen3.8-27B, not established by executing tests. [2,5]')
sub('What is established')
p('The LoRA capacity and attachment sites are unchanged. A fresh base already produces reproducible GPU-dependent logits. The old trained adapter turns those differences into strong A/B separation on A100/RTX in BF16; the same stored weights lose this separation under FP32. Activation-transfer interventions demonstrate a causal role for the differing residual states.')
sub('What is a supported explanation, not yet a training ablation')
p('Whole-solution teacher forcing, token averaging, varying prompts, physical batch shapes and checkpoint selection by continuation loss plausibly make the coding router harder to learn. The objective analysis and observed shared-language outputs make this a concrete explanation, but do not prove which factor dominates. The data-quality examples may impede coding correctness; their effect on hardware routing was not isolated.')
sub('How much should we infer from the earlier “good performance”?')
p('Run-003\'s selected step 750 gave 100% greedy accuracy across the two trained hardware cases, with mean answer-token loss 0.00384358. Selection used the same one trained prompt. At terminal step 950, both GPUs answered A and mean loss regressed to 0.74758808. Therefore the strong result is a selected checkpoint observation, not stable convergence or held-out prompt accuracy. The new A100 transfer is useful additional evidence, but one prompt on one extra device cannot establish a general GPU-identity detector. [3,4]')
sub('Follow-up experiments worth deciding on separately')
p('First, measure first-divergence losses and language-choice margins alongside full-solution loss, without changing training. Second, run a controlled matched-prompt comparison to separate the effects of prompt diversity, physical batching and cache mode. Third, if the goal is specifically routing, compare the current objective with explicitly increased weight on the first conflicting decision. These are proposals, not changes applied to the completed experiment.')

page();h('8. Provenance, verification and references')
p('The primary experiment comprises two weight states x three numerical settings x two GPUs, with three repeat forwards per condition. Additional hooked forwards, adapter-disable controls, and 76 donor plus 76 self activation patches test interpretation. All completed probe and patch stages exited 0. Initial patch launches reached the donor-file read before transfers finished; those failed before model execution, were preserved, and were retried only after the donor files existed and the GPUs were idle. All completed arrays passed manifest size/SHA256 checks. [2]')
table([['Identity','Value'],
 ['Original base revision','aa8e72537993ba99e69dfaafa59ed015b17504d1'],
 ['Frozen-base tensor SHA256','7fdd4eeb373e7b8c86baa9cfd230a22a8f237780a25820476b0c3a00bb1e5525'],
 ['Published adapter revision','ca027b140a4889357d5fdde6495da9f5697e8019'],
 ['Adapter safetensors SHA256','d8695f3d07464e3f1ad127b3f12de0e0d46b235d6f2ebbfe2207ea1a940343a1'],
 ['Raw measurements and exact prompts','outputs/numerical-probes/20261023/ (task workspace; excluded from Git)'],
 ['Source and compact measurements','analysis/numerical_probe_20261023/ and results/numerical-probe-20261023/ in the gpu-awareness repository'],
 ],[1.9*inch,5.2*inch])
p('Raw artifacts include complete next-token logit vectors, all-position residual states, manifests, process logs and exact first-update coding prompts. The report and figures are static artifacts. No W&B run or dashboard was changed, no old experiment was restarted, and no new GPU was rented. Both Pods were left running. All new Git files were checked against the strict 20,000,000-byte limit.','SmallReport')
sub('Source references')
refs=[
 ('1','Coding configuration and loader', 'https://github.com/FlyingPumba/gpu-awareness/blob/033e1b9/coding2/config.json',
  'Also coding2/model.py and coding2/train.py at the same repository revision; actual coding run provenance verifies the parameter layout and frozen-base hash.'),
 ('2','New measurements and reproducible analysis', 'https://github.com/FlyingPumba/gpu-awareness/tree/033e1b9/analysis/numerical_probe_20261023',
  'Compact results: results/numerical-probe-20261023/analysis.json. Raw tensors, process records and exact prompts are stored locally under the path above.'),
 ('3','Original one-token recipe and forward evaluation', 'https://github.com/FlyingPumba/gpu-awareness/blob/033e1b9/configs/run-003-qwen25-3b.json',
  'See train.py and results/run-003-qwen25-3b/ for the archived evaluation and completion records.'),
 ('4','Published GPU-awareness model, pinned commit', 'https://huggingface.co/iarcuschin/gpu-awareness-Qwen2.5-3B-Instruct/tree/ca027b140a4889357d5fdde6495da9f5697e8019',
  'Hub API identity and adapter LFS hash were checked; the original BF16 base and FP32 adapter were loaded separately.'),
 ('5','Completed coding-002 report', 'https://github.com/FlyingPumba/gpu-awareness/blob/033e1b9/coding2/results/coding-002-qwen25-3b/RESULTS.txt',
  'Final language, correctness, truncation and checkpoint-selection results. Paired language counts were recomputed from the preserved judgments for this report.'),
 ('6','Official pinned Qwen architecture configuration', 'https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/blob/aa8e72537993ba99e69dfaafa59ed015b17504d1/config.json',
  'Layer count, dimensions, attention heads and tied embeddings.'),
 ('7','PyTorch 2.8 numerical accuracy notes', 'https://docs.pytorch.org/docs/2.8/notes/numerical_accuracy.html',
  'Floating-point operations need not be bitwise identical across devices or batched shapes; lower-precision reductions can change results.'),
 ('8','PyTorch 2.8 scaled-dot-product attention', 'https://docs.pytorch.org/docs/2.8/generated/torch.nn.functional.scaled_dot_product_attention.html',
  'Backend selection and fused-operation rounding can change attention outputs. Exact CUDA kernels were not profiled in this study.'),
]
for n,title,url,note in refs:
    p(f'[{n}] <link href="{url}" color="#176b8a"><u>{title}</u></link>. {escape(note)}','SmallReport')


def footer(canvas,doc):
    canvas.saveState();canvas.setStrokeColor(colors.HexColor('#c8d1da'))
    canvas.line(.6*inch,.46*inch,7.7*inch,.46*inch)
    canvas.setFont('Helvetica',8);canvas.setFillColor(colors.HexColor('#5a6e7d'))
    canvas.drawString(.6*inch,.29*inch,'Qwen coding SFT and GPU numerical awareness | 23 October 2026')
    canvas.drawRightString(7.7*inch,.29*inch,str(doc.page));canvas.restoreState()

dest=OUT/'REPORT.pdf'
doc=SimpleDocTemplate(str(dest),pagesize=(8.3*inch,11.7*inch),
                      leftMargin=.6*inch,rightMargin=.6*inch,topMargin=.55*inch,bottomMargin=.65*inch,
                      title='Coding SFT and GPU numerical awareness',author='GPU awareness experiment analysis')
doc.build(story,onFirstPage=footer,onLaterPages=footer)
print(dest)
