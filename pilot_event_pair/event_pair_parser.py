"""Conservative English two-event grammar for the CPU pilot."""
import re

from observable_cue_probe import VERBS


EXTRA = '''look looks looked looking touch touches touched touching watch watches watched watching
press presses pressed pressing lean leans leaned leaning squat squats squatted squatting
pose poses posed posing stay stays stayed staying kiss kisses kissed kissing
cycle cycles cycled cycling nod nods nodded nodding stare stares stared staring
chase chases chased chasing bite bites bit biting sniff sniffs sniffed sniffing
feed feeds fed feeding hug hugs hugged hugging toss tosses tossed tossing
grasp grasps grasped grasping grab grabs grabbed grabbing bend bends bent bending
move moves moved moving go goes went going come comes came coming arrive arrives arrived arriving
turn turns turned turning start starts started starting finish finishes finished finishing
stop stops stopped stopping leave leaves left leaving enter enters entered entering
return returns returned returning approach approaches approached approaching follow follows followed following
drive drives drove driving steer steers steered steering wave waves waved waving
raise raises raised raising lower lowers lowered lowering step steps stepped stepping
flip flips flipped flipping fall falls fell falling jump jumps jumped jumping
join joins joined joining cross crosses crossed crossing land lands landed landing
smell smells smelled smelling brush brushes brushed brushing peel peels peeled peeling
mix mixes mixed mixing fold folds folded folding eat eats ate eating drink drinks drank drinking
cook cooks cooked cooking bake bakes baked baking stack stacks stacked stacking
place places placed placing bring brings brought bringing hand hands handed handing
switch switches switched switching set sets setting take takes took taking
give gives gave giving get gets got getting make makes made making
lift lifts lifted lifting inspect inspects inspected inspecting check checks checked checking
point points pointed pointing smile smiles smiled smiling laugh laughs laughed laughing
talk talks talked talking speak speaks spoke speaking shout shouts shouted shouting
scream screams screamed screaming turn turns turned turning clap claps clapped clapping
pat pats patted patting pet pets petted petting clean cleans cleaned cleaning
touch touches touched touching tap taps tapped tapping step steps stepped stepping
kick kicks kicked kicking pound pounds pounded pounding combine combines combined combining
clasp clasps clasped clasping count counts counted counting signal signals signaled signaling
assist assists assisted assisting snuggle snuggles snuggled snuggling rest rests rested resting
remove removes removed removing wear wears wore wearing untie unties untied untying
refuse refuses refused refusing lick licks licked licking wag wags wagged wagging
fold folds folded folding unfold unfolds unfolded unfolding stumble stumbles stumbled stumbling
trip trips tripped tripping balance balances balanced balancing rub rubs rubbed rubbing
chop chops chopped chopping dig digs dug digging dip dips dipped dipping sink sinks sank sinking
catch catches caught catching reach reaches reached reaching duck ducks ducked ducking
pack packs packed packing unpack unpacks unpacked unpacking tie ties tied tying
play plays played playing practice practices practiced practicing perform performs performed performing'''
ACTION = set(VERBS) | set(EXTRA.split())
AUX = {'is','are','was','were','has','have','had','begins','began','first','finally','just'}
PRONOUNS = {'he','she','it','they'}
REJECT = {'not','never','without','nobody','nothing','neither','nor'}
Q = re.compile(r'^(?:what (?:did|do|does) (.+?) do|how (?:did|do|does) (.+?) react) (before|after) (.+)$')
NOUN_START = {'a','an','the','this','that','another','one','some','two','three','his','her','their'} | PRONOUNS


def normalize(text):
    return ' '.join(re.findall(r"[a-z0-9]+",text.lower()))


def parse(question, choices):
    q = normalize(question)
    if len(re.findall(r'\b(?:before|after)\b',q))!=1:
        return None
    m = Q.fullmatch(q)
    if not m:
        return None
    actor = m[1] or m[2]
    relation = m[3]
    tail = m[4].split()
    if not tail or actor in PRONOUNS or actor in {'someone','people'} or set(tail)&REJECT:
        return None
    first = tail[0]
    if first.endswith('ing') and first in ACTION:
        anchor = f'{actor} {m[4]}'
    elif first in NOUN_START:
        # Detect the predicate after an explicit subject or a pronoun resolved to the actor.
        if not any(w in ACTION for w in tail[1:7]):
            return None
        anchor = (actor+' '+' '.join(tail[1:])) if first in PRONOUNS else m[4]
    else:
        return None
    targets = []
    for choice in choices:
        words = normalize(choice).split()
        if not words or set(words)&(REJECT|{'before','after'}):
            return None
        if words[0]=='to':
            words=words[1:]
        if not words or not (words[0] in ACTION or words[0] in NOUN_START and any(w in ACTION for w in words[1:6])):
            return None
        subject = '' if words[0] in NOUN_START else actor+' '
        targets.append(subject+' '.join(words))
    if len(set(targets))!=5 or anchor in targets:
        return None
    return (relation,anchor,*targets)
