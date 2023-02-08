from pyChatGPT import ChatGPT
from youtubesearchpython import VideosSearch
import shutil
import re
import os
import sys
import time
import random
import vlc
import torch
import numpy as np
sys.path.append('tacotron2/')
sys.path.append('waveglow/')

from hparams import create_hparams
from model import Tacotron2
from layers import TacotronSTFT, STFT
from audio_processing import griffin_lim
from train import load_model
from text import text_to_sequence
from denoiser import Denoiser
import soundfile as sf
import simpleaudio as sa
from pydub import AudioSegment 
from pydub.playback import play
import threading
import tkinter as tk

organisms = ["Quercus alba", "Pinus sylvestris", "Acer rubrum", "Populus tremuloides", "Betula papyrifera", "Picea abies", "Abies balsamea", "Salix babylonica", "Populus balsamifera", "Alnus incana", "Fraxinus americana", "Castanea sativa", "Juglans nigra", "Liriodendron tulipifera", "Tsuga canadensis", "Ginkgo biloba", "Carya ovata", "Sassafras albidum", "Ulmus americana", "Fagus grandifolia", "Cercis canadensis", "Rhododendron maximum", "Sorbus americana", "Carpinus caroliniana", "Prunus serotina", "Crataegus crus-galli", "Ostrya virginiana", "Tilia americana", "Malus pumila", "Hippophae rhamnoides", "Euonymus americanus", "Corylus americana", "Amelanchier arborea", "Pyrus communis", "Acer saccharum", "Rhamnus cathartica", "Zelkova serrata", "Diospyros virginiana", "Lonicera japonica", "Sambucus canadensis", "Viburnum dentatum", "Viburnum opulus", "Vitis vinifera", "Vaccinium corymbosum", "Arbutus unedo", "Cornus sericea", "Cornus stolonifera", "Ilex opaca", "Quercus robur", "Aesculus hippocastanum", "Liriodendron tulipifera", "Cedrus deodara", "Cedrus atlantica", "Pinus wallichiana", "Pinus ponderosa", "Pinus cembra", "Pinus mugo", "Pinus rigida", "Pinus sylvestris var. hamata", "Pinus sylvestris var. mughus", "Pinus sylvestris var. mongolica", "Pinus banksiana", "Pinus resinosa", "Pinus contorta", "Pinus parviflora", "Pinus nigra", "Pinus canariensis", "Pinus halepensis", "Pinus pinaster", "Pinus densiflora", "Pinus pinea", "Pinus peuce", "Pinus jeffreyi", "Pinus longaeva", "Pinus coulteri", "Pinus torreyana", "Pinus radiata", "Pinus leiophylla", "Pinus monophylla", "Pinus strobus", "Pinus ponderosa var. scopulorum", "Pinus strobus var. chiapasensis", "Pinus strobus var. strobus", "Pinus strobus var. ecuadoriensis", "Pinus strobus var. neostrobus", "Pinus strobus var. pendula", "Pinus strobus var. pseudostrobus", "Pinus strobus var. strobiformis", "Pinus strobus var. virginiana", "Pinus virginiana", "Pinus echinata", "Pinus clausa", "Pinus palustris", "Pinus serotina", "Pinus virginiana var. virginiana", "Homo sapiens", "Canis lupus", "Felis catus", "Panthera leo", "Equus caballus", "Bos taurus", "Ovis aries", "Cervus elaphus", "Sus scrofa", "African Elephant", "Lion", "Giraffe", "Crocodile", "Hippopotamus", "Rhino", "Gorilla", "Cheetah", "Baboon", "Buffalo", "Kudu", "Zebra", "Leopard", "Hyena", "Wildebeest", "Impala", "Antelope", "Hartebeest", "Eland", "Warthog", "Ostrich", "Chimpanzee", "Jaguar", "Monkey", "Nyala", "Ocelot", "Civet", "Bushbaby", "Baboon Spider", "Bison", "Chacma Baboon", "Common Duiker", "Cape Eland", "Giant Forest Hog", "Sable Antelope", "Buffalo Weaver", "Black Rhinoceros", "Red Duiker", "Gazelle", "Vervet Monkey", "African Wild Dog", "Lioness", "Common Waterbuck", "Plains Zebra", "Klipspringer", "Yellow-Backed Duiker", "Reedbuck", "Spotted Hyena", "African Elephant Calf", "African Buffalo Calf", "African Elephant Family", "Leopard Cub", "Lion Cub", "Giraffe Calf", "Lioness and Cubs", "Lion Pride", "Chimpanzee Family", "Gorilla Family", "Rhino Calf", "Hippo Calf", "Baboon Troop", "African Wild Dog Pack", "Giraffe Herd", "Elephant Herd", "Zebra Herd", "Antelope Herd", "Lioness Hunting", "Gorilla Eating", "Rhino Drinking", "Hippo Wallowing", "Chimpanzee Playing", "Lion Roaring", "Giraffe Stretching", "Cheetah Running", "African Elephant Trunk", "Lion Mane", "Giraffe Neck", "Crocodile Mouth", "Hippopotamus Teeth", "Rhino Horn", "Gorilla Chest", "Cheetah Spot", "Baboon Bottom", "Buffalo Shoulder", "Kudu Horns", "Zebra Stripes", "Leopard Print", "Hyena Laugh", "Wildebeest Migration", "Impala Jump", "Antelope Grazing", "Hartebeest Charge", "Eland Charge", "Warthog Tusk", "Ostrich Feather", "Chimpanzee Smile", "Jaguar Roar", "Monkey Swing", "Nyala Horns", "Ocelot Whiskers", "Civet Tail", "Bushbaby Eyes", "Baboon Spider Web", "Bison Shoulder", "Chacma Baboon Face", "Common Duiker Horns", "Cape Eland Antlers", "Giant Forest Hog Tusk", "Sable Antelope Horns", "Buffalo Weaver Nest", "Black Rhinoceros Horn", "Red Duiker Horns", "Gazelle Hooves", "Vervet Monkey Call", "Asian Elephant", "Bengal Tiger", "Indian Rhinoceros", "Himalayan Bear", "Gaur", "Wild Water Buffalo", "Snow Leopard", "Sambar Deer", "Kouprey", "Hippopotamus", "Asiatic Lion", "Goral", "Nilgai", "Civet", "Sloth Bear", "Banteng", "Indian Mongoose", "Barbary Macaque", "Golden Langur", "Clouded Leopard", "Black-tailed Deer", "Indian Grey Mongoose", "Lar Gibbon", "Indian Pangolin", "Malabar Giant Squirrel", "Grizzled Giant Squirrel", "Bharal", "Muntjac", "Marbled Cat", "Small Indian Civet", "Striped Hyena", "Jungle Cat", "Serow", "Leopard Cat", "Asian Black Bear", "Brown Palm Civet", "Large-tailed Nightjar", "Yellow-throated Marten", "Pygmy Hog", "Indian Wolf", "Red Panda", "Bengal Fox", "Greater One-horned Rhinoceros", "Lesser One-horned Rhinoceros", "Nepal Grey Langur", "Hog Deer", "Spotted Linsang", "Chestnut-bellied Sandgrouse", "Blyth's Tragopan", "Chital", "Sri Lankan Leopard", "Sri Lankan Sloth Bear", "Sri Lankan Elephant", "Golden Jackal", "Bengal Slow Loris", "Plains Zebra", "Hornbill", "Pangolin", "Wild Boar", "Indian Peafowl", "Pelican", "Brahminy Kite", "Osprey", "Ibis", "Eagle", "Stork", "Heron", "Kingfisher", "Pheasant", "Parakeet", "Flamingo", "Hornbilled Duck", "Sparrow", "Robin", "Pigeon", "Dove", "Peacock", "Crow", "Owl", "Falcon", "Vulture", "Kite", "Swan", "Duck", "Herring Gull", "Cormorant", "Seagull", "Tern", "Osprey", "Kestrel", "Hawk", "Eagle Owl", "Eagle-owl", "Long-tailed Shrike", "Magpie", "Cuckoo", "Drongo", "Thrush", "Starling", "Lark", "Sunbird", "Bee-eater", "Kingfisher", "Green Pigeon", "Parrot", "Woodpecker", "Nightjar", "Swift", "Wagtail", "Finch", "Pelican", "Bustard", "Shorebird", "Sandpiper", "Plover", "Lapwing", "Stilt", "Dotterel", "Teal", "Pintail", "Pochard", "Mallard", "Gadwall", "Wigeon", "Shoveler", "Pochard", "Goose", "Swan", "Crane", "Stork", "Heron", "Ibis", "Egret", "Bittern", "Spoonbill", "Elephas maximus", "Bos gaurus", "Bos javanicus", "Bos frontalis", "Bubalus bubalis", "Bubalus arnee", "Cervus unicolor", "Cervus nippon", "Sus scrofa", "Sus barbatus", "Suidae sp.", "Capricornis sumatraensis", "Hippopotamus amphibius", "Hippopotamus amphibius variegatus", "Rhinoceros unicornis", "Rhinoceros sondaicus", "Dicerorhinus sumatrensis", "Tursiops aduncus", "Orcinus orca", "Ganges River Dolphin", "Lagenodelphis hosei", "Platanista gangetica", "Platanista minor", "Vicugna pacos", "Lama glama", "Lama guanicoe", "Alpaca Pacos", "Ovis aries", "Capra hircus", "Nemorhaedus goral", "Nemorhaedus sumatraensis", "Serow Nemorhaedus", "Thinhorn Sheep", "Ovis ammon", "Antilope cervicapra", "Antilope saiga", "Gazella subgutturosa", "Gazella bennettii", "Gazella gazella", "Panthera tigris", "Panthera pardus", "Panthera onca", "Panthera leo", "Jaguar Panthera", "Leopardus pardalis", "Leopardus wiedii", "Snow Leopard", "Leopardus uncia", "Felis bengalensis", "Felis silvestris", "Felis manul", "Felis margarita", "Canis lupus", "Canis aureus", "Canis rufus", "Ursus arctos", "Ursus thibetanus", "Asiatic Black Bear", "Ursus malayanus", "Ursus maritimus", "Pinnipedia", "Otariidae", "Otaria flavescens", "Zalophus californianus", "Eumetopias jubatus", "Phocidae", "Erignathus barbatus", "Leptonychotes weddellii", "Mirounga angustirostris", "Halichoerus grypus", "Lobodon carcinophaga", "Hydrurga leptonyx", "Crabeater Seal", "Leopard Seal", "Ommatophoca rossii", "Ross Seal", "Phocarctos hookeri", "Arctocephalus forsteri", "Arctocephalus galapagoensis", "Arctocephalus philippii", "Arctocephalus townsendi", "Lontra canadensis", "Lontra felina", "Mustela lutreola", "Mustela erminea", "Mustela nivalis", "Mustela putorius", "Martes zibellina", "Martes americana", "Viverridae", "Viverra zibetha", "Viverra civettina", "Viverra tangalunga", "Herpestidae", "Herpestes javanicus", "Herpestes smithii", "Herpestes ichneumon", "Cynictis penicillata", "Abelmoschus moschatus", "Acorus calamus", "Adiantum capillus-veneris", "Aegle marmelos", "Aesculus turbinata", "Alocasia macrorrhiza", "Alpinia galanga", "Alpinia oxyphylla", "Amla", "Anacardium occidentale", "Anemarrhena asphodeloides", "Angelica dahurica", "Antennaria dioica", "Artemisia annua", "Artemisia apiacea", "Artemisia capillaris", "Artemisia japonica", "Artemisia vulgaris", "Asarum sieboldii", "Asclepias syriaca", "Asparagus racemosus", "Atractylodes lancea", "Bambusa arundinacea", "Bambusa tulda", "Berberis aquifolium", "Bergenia crassifolia", "Betula platyphylla", "Bischofia javanica", "Bixa orellana", "Borago officinalis", "Boswellia carterii", "Brassica rapa", "Broussonetia papyrifera", "Camellia sinensis", "Canna indica", "Capsicum annuum", "Carica papaya", "Carum carvi", "Carya illinoinensis", "Carya ovata", "Carya sinensis", "Castanea mollissima", "Cedrella sinensis", "Celastrus orbiculatus", "Centella asiatica", "Cercis chinensis", "Chamaemelum nobile", "Chrysanthemum coronarium", "Cinnamomum camphora", "Cinnamomum verum", "Cistanche deserticola", "Cistus incanus", "Citrullus lanatus", "Clinopodium acinos", "Clitoria ternatea", "Cocos nucifera", "Commiphora myrrha", "Coptis chinensis", "Cornus officinalis", "Crataegus pinnatifada", "Cryptomeria japonica", "Cucurbita pepo", "Cupressus funebris", "Cupressus sempervirens", "Cydonia oblonga", "Cymbopogon citratus", "Cyperus rotundus", "Daucus carota", "Dendrocalamus giganteus", "Dendrobium candidum", "Dendrobium fimbriatum", "Dendrobium nobile", "Dendrobium ochreatum", "Desmodium gangeticum", "Diospyros kaki", "Dolichos lablab", "Echinacea purpurea", "Elaeagnus angustifolia", "Elaeis guineensis", "Eleutherococcus senticosus", "Emblica officinalis", "Erythrina variegata", "Eucalyptus globulus", "Eucommia ulmoides", "Ficus carica", "Ficus religiosa", "Garcinia cambogia", "Glycyrrhiza glabra", "Gossypium hirsutum", "Gynura segetum", "Hedera helix", "Hibiscus sabdariffa", "Hypericum perforatum", "Ilex paraguariensis", "Illicium verum", "Indigofera tinctoria", "Jasminum officinale", "Juglans regia", "Juniperus chinensis", "Kadsura japonica", "Laminaria japonica", "Larix gmelinii", "Lavandula angustifolia", "Lepidium meyenii", "Lindera aggregata", "Lonicera japonica", "Lycium barbarum", "Magnolia kobus", "Magnolia officinalis", "Mangifera indica", "Mentha piperita", "Metasequoia glyptostroboides", "Micromelum pubescens", "Morus alba", "Morus nigra", "Nelumbo nucifera", "Nepeta cataria", "Olea europaea", "Oroxylum indicum", "Panax ginseng", "Panicum miliaceum", "Papaver somniferum", "Paradisea liliastrum", "Paeonia lactiflora", "Pelargonium graveolens", "Pelargonium x hortorum", "Perilla frutescens", "Picea abies", "Pinus armandii", "Pinus densiflora", "Pinus massoniana", "Pinus sylvestris", "Pistacia chinensis", "Pistacia vera", "Pleurotus ostreatus", "Podocarpus macrophyllus", "Polygonum multiflorum", "Prunus avium", "Prunus cerasus", "Prunus dulcis", "Prunus persica", "Punica granatum", "Raphanus sativus", "Rhamnus cathartica", "Rhododendron indicum", "Ricinus communis", "Robinia pseudoacacia", "Rosa chinensis", "Rosa rugosa", "Rubus idaeus", "Salix babylonica", "Sambucus nigra", "Sapindus mukorossi", "Sassafras albidum", "Schisandra chinensis", "Scutellaria baicalensis", "Sesamum indicum", "Silybum marianum", "Smilax glabra", "Sophora flavescens", "Sophora japonica", "Soymida febrifuga", "Styrax japonicus", "Swertia japonica", "Syzygium aromaticum", "Taxus chinensis", "Taxus cuspidata", "Thymus vulgaris", "Tilia cordata", "Tinospora cordifolia", "Toona sinensis", "Ulmus parvifolia", "Urtica dioica", "Vaccinium myrtillus", "Vaccinium oxycoccos", "Vaccinium uliginosum", "Vitis vinifera", "Zingiber officinale", "Acanthocalycium klimpelianum", "Acerola", "Alchornea castaneifolia", "Amaranthus hypochondriacus", "Annona cherimola", "Anthurium andraeanum", "Aralia elata", "Ardisia elliptica", "Arrabidaea chica", "Arum maculatum", "Aspidosperma cylindrocarpon", "Astrocaryum murumuru", "Attalea phalerata", "Bactris gasipaes", "Bauhinia forficata", "Berberis darwinii", "Bixa orellana", "Bourreria ovata", "Brachychiton populneus", "Bromelia karatas", "Brunfelsia grandiflora", "Cabralea canjerana", "Cajanus cajan", "Calathea lutea", "Calathea zebrina", "Calotropis gigantea", "Calyptranthes concinna", "Calyptranthes pauciflora", "Calyptranthes spinosa", "Calyptrogyne ghiesbreghtiana", "Calyxochoma spectabilis", "Camellia sinensis", "Campomanesia xanthocarpa", "Cananga odorata", "Capsicum annuum", "Carica papaya", "Caryocar villosum", "Cecropia obtusifolia", "Cestrum aurantiacum", "Chrysophyllum cainito", "Cissus verticillata", "Clausena anisata", "Clusia rosea", "Coccoloba uvifera", "Coffea arabica", "Columnea xanthiifolia", "Conostegia xalapensis", "Copernicia alba", "Copernicia baileyana", "Copernicia cerifera", "Copernicia prunifera", "Copernicia tectorum", "Cordia goeldiana", "Cordia alliodora", "Costus spicatus", "Couma macrocarpa", "Crataegus monogyna", "Cryptocarya alba", "Cupania americana", "Cupressus sempervirens", "Cyathea medullaris", "Cyathea lepifera", "Dendropanax cuneatus", "Dendropanax morototoni", "Dendropanax trifidus", "Desmanthus virgatus", "Dimorphandra mollis", "Dimorphotheca sinuata", "Dioscorea alata", "Drymis winteri", "Duschekia fruticosa", "Eichhornia crassipes", "Elettaria cardamomum", "Embelia ribes", "Enterolobium cyclocarpum", "Eryngium foetidum", "Erythrina crista-galli", "Eschweilera coriacea", "Eucalyptus citriodora", "Eucalyptus grandis", "Eucalyptus marginata", "Eucalyptus regnans", "Euphorbia heterophylla", "Euterpe edulis", "Ficus benjamina", "Ficus elastica", "Ficus retusa", "Garcinia dulcis", "Ginger", "Gmelina arborea", "Guazuma ulmifolia", "Heteropterys auriculata", "Hibiscus rosa-sinensis", "Hydrocotyle ranunculoides", "Hymenaea courbaril", "Inga edulis", "Ipomoea batatas", "Jacaranda mimosifolia", "Juglans regia", "Juniperus communis", "Lantana camara", "Litchi chinensis", "Lonchocarpus nitidus", "Lucuma salicifolia", "Malpighia glabra", "Mangifera indica", "Manihot esculenta", "Matayba elaeagnoides", "Mauritia flexuosa", "Melaleuca leucadendron", "Melastoma malabathricum", "Melastomataceae", "Mitracarpus villosus", "Myrciaria dubia", "Myrtaceae", "Nectandra rodiaei", "Nectandra sanguinea", "Ocimum basilicum", "Orchidaceae", "Oreopanax daphnifolius", "Oreopanax xalapensis", "Oryza sativa", "Oxandra lanceolata", "Pachira aquatica", "Paederia scandens", "Pancratium maritimum", "Peperomia pellucida", "Persea americana", "Petiveria alliacea", "Phaseolus lunatus", "Philodendron bipinnatifidum", "Philodendron scandens", "Physalis peruviana", "Picea abies", "Pimenta dioica", "Pinus radiata", "Piper auritum", "Piper nigrum", "Pitcairnia heterophylla", "Pleiostachya pruinosa", "Podocarpus nagi", "Polypodiaceae", "Pouteria campechiana", "Pouteria sapota", "Prunus avium", "Psidium guajava", "Pterocarpus rohrii", "Punica granatum", "Rheedia edulis", "Rosa canina", "Rosmarinus officinalis", "Rubus fruticosus", "Sambucus nigra", "Sapindus saponaria", "Sapium glandulosum", "Sassafras albidum", "Scutia buxifolia", "Serenoa repens", "Solanum lycopersicum", "Solanum quitoense", "Solanum tuberosum", "Sterculia apetala", "Swietenia mahagoni", "Symplocos cochinchinensis", "Talauma ovata", "Tecoma stans", "Tithonia diversifolia", "Tocoyena formosa", "Tropaeolum majus", "Ulex europaeus", "Urtica dioica", "Vaccinium meridionale", "Vernonia polyanthes", "Viburnum tinus", "Vochysia guatemalensis", "Wisteria floribunda", "Zea mays", "Amazona aestiva", "Andean Condor", "Vultur gryphus", "Anhima cornuta", "Southern Screamer", "Rhea americana", "Ovis aries", "Lama glama", "Alpaca pacos", "Vicugna pacos", "Ceratotherium simum", "Elephantulus rufescens", "Hydrochoerus hydrochaeris", "Myocastor coypus", "Otaria flavescens", "Callithrix jacchus", "Callicebus moloch", "Saguinus geoffroyi", "Saimiri boliviensis", "Ateles belzebuth", "Lagothrix lagotricha", "Cacajao calvus", "Aotus nancymaae", "Cavia porcellus", "Dasyprocta aguti", "Agouti paca", "Geocapromys brownii", "Tayassu pecari", "Dasypus novemcinctus", "Bradypus variegatus", "Cougar", "Puma concolor", "Jaguar", "Panthera onca", "Ocelot", "Leopardus pardalis", "Oncifelis colocolo", "Felis concolor", "Panthera tigris", "Lion", "Acinonyx jubatus", "Cheetah", "Eira barbara", "Tayassu tajacu", "Sus scrofa", "Wild Boar", "Lama guanicoe", "Guanaco", "Vicugna vicugna", "Llama", "Dama dama", "Fallow Deer", "Odocoileus virginianus", "Alces alces", "Moose", "Cervus elaphus", "Red Deer", "Rangifer tarandus", "Reindeer", "Procyon lotor", "Raccoon", "Ursus americanus", "Black Bear", "Potos flavus", "Kinkajou", "Nasua nasua", "Coati", "Galeopterus variegatus", "Flying Squirrel", "Sciurus carolinensis", "Tamandua tetradactyla", "Anteater", "Conepatus semistriatus", "Skunk", "Myotis velifer", "Big Brown Bat", "Lasionycteris noctivagans", "Silver-haired Bat", "Eptesicus fuscus", "Big Brown Bat", "Lasiurus cinereus", "Hoary Bat", "Phyllostomus hastatus", "Pallas's Long-tongued Bat", "Chiroptera", "Bats", "Eumops perotis", "Hooded Bat", "Glossophaga soricina", "Common Long-tongued Bat", "Didelphis marsupialis", "Common Opossum", "Conepatus chinga", "Andean Skunk", "Conepatus leuconotus", "White-backed Skunk", "Mephitis mephitis", "Striped Skunk", "Spilogale putorius", "Skunk", "Chrysocyon brachyurus", "Maned Wolf", "Canis latrans", "Coyote", "Canis lupus", "Gray Wolf", "Canis simensis", "Lycalopex gymnocercus", "Patagonian Fox", "Pseudalopex culpaeus", "Darwin's Fox", "Lontra canadensis", "North American River Otter", "Lontra longicaudis", "Southern River Otter", "Eira barbara", "Tayassu tajacu", "Sus scrofa", "Wild Boar", "Felis concolor", "Panthera tigris", "Lion", "Acinonyx jubatus", "Cheetah", "Hydrochoerus hydrochaeris", "Myocastor coypus", "Otaria flavescens", "Callithrix jacchus", "Callicebus moloch", "Saguinus geoffroyi", "Saimiri boliviensis", "Ateles belzebuth", "Lagothrix lagotricha", "Cacajao calvus", "Aotus nancymaae", "Cavia porcellus", "Dasyprocta aguti", "Agouti paca", "Geocapromys brownii", "Tayassu pecari", "Dasypus novemcinctus", "Bradypus variegatus", "Vultur gryphus", "Andean Condor", "Cervus elaphus", "Red Deer", "Rangifer tarandus", "Reindeer", "Alces alces", "Moose", "Odocoileus virginianus", "Fallow Deer", "Lama glama", "Alpaca pacos", "Vicugna pacos", "Ceratotherium simum", "Elephantulus rufescens", "Ovis aries", "Rhea americana", "Southern Screamer", "Anhima cornuta", "Amazona aestiva", "Phyllostomus hastatus", "Pallas's Long-tongued Bat", "Chiroptera", "Bats", "Eumops perotis", "Hooded Bat", "Glossophaga soricina", "Common Long-tongued Bat", "Lasionycteris noctivagans", "Silver-haired Bat", "Eptesicus fuscus", "Big Brown Bat", "Lasiurus cinereus", "Hoary Bat", "Myotis velifer", "Big Brown Bat", "Mephitis mephitis", "Striped Skunk", "Spilogale putorius", "Skunk", "Conepatus leuconotus", "White-backed Skunk", "Conepatus chinga", "Andean Skunk", "Didelphis marsupialis", "Common Opossum", "Nasua nasua", "Coati", "Potos flavus", "Kinkajou", "Ursus americanus", "Black Bear", "Procyon lotor", "Raccoon", "Tamandua tetradactyla", "Anteater", "Sciurus carolinensis", "Flying Squirrel", "Galeopterus variegatus", "Anteater", "Cougar", "Puma concolor", "Jaguar", "Panthera onca", "Ocelot", "Leopardus pardalis", "Oncifelis colocolo", "Canis lupus", "Gray Wolf", "Canis latrans"]

class Control:
    def __init__(self):
        self.is_audio_playing = False
        self.is_scene_running = False
        self.is_inference_ready = False
        self.is_inference_running = False
        self.get_video_ready = False
        self.get_video_running = False
        self.get_chatgpt_text_ready = False
        self.get_chatgpt_text_running = False
        self.is_play_audio_running = False
        self.keyword_reset = True
        self.chatgpt_message = ""
        self.keyword = ""
        self.wav_list = []
        self.vlc_obj = vlc.Instance("--no-xlib")
        self.vlcplayer = self.vlc_obj.media_player_new()
        self.label = None
        self.title_label = None
        self.video_title = ""

        self.start_time = 0

        self.taco_model = None
        self.waveglow_model = None
        self.search_text = ""
        self.taco_model_path = "attenborough_v4_checkpoint_4225700(5050500)"
        self.waveglow_model_path = "attenborough_v4_waveglow_2030200(3133000)1e-5"

        hparams = create_hparams()
        hparams.sampling_rate = 22050
        hparams.p_attention_dropout = 0
        hparams.p_decoder_dropout = 0
        hparams.max_decoder_steps = 2000

        self.taco_model = load_model(hparams)
        self.taco_model.load_state_dict(torch.load(self.taco_model_path)['state_dict'])
        _ = self.taco_model.cuda().eval().half()

        self.waveglow_model = torch.load(self.waveglow_model_path)['model']
        self.waveglow_model.cuda().eval().half()

    def start_timer(self):
        print("starting timer at {}".format(time.time()))
        self.start_time = time.time()  

    def run(self):
        while True:
            root.update_idletasks()
            root.update()
                    
            if self.keyword_reset:
                i = len(organisms)
                self.keyword = organisms[random.randint(0, i-1)]
                self.keyword_reset = False            

            if not self.get_video_ready and not self.get_video_running:
                self.get_video() #threaded
            
            if not self.get_chatgpt_text_ready and not self.get_chatgpt_text_running:
                self.get_chatgpt_text("a 5 paragraph text on {}, narrated by david attenborough. use some funny nature jokes.".format((self.keyword).strip())) #threaded

            if not self.is_inference_ready and self.get_chatgpt_text_ready and not self.is_inference_running:
                self.run_inference() #threaded

            if self.get_video_ready and self.get_chatgpt_text_ready and self.is_inference_ready and not self.is_play_audio_running:                
                self.play_audio() # threaded
                time.sleep(2)
                self.play_main_video()      
            time.sleep(1)
            
    def play(self):
        #copy ready video into main video
        shutil.copyfile("prep_video.mp4", "main_video.mp4")
        
        #copy new wavs
        for i, p in enumerate(self.wav_list):
            shutil.copyfile(p, "wav_{}.wav".format(i))

        self.is_audio_playing = True

        time.sleep(3)

        for i, w in enumerate(self.wav_list):
            wave_obj = sa.WaveObject.from_wave_file("wav_{}.wav".format(i))            
            play_obj = wave_obj.play()
            play_obj.wait_done()
        self.is_audio_playing = False
        #restart setup
        while True:
            if self.get_chatgpt_text_ready and self.get_video_ready and self.is_inference_ready:
                break
            time.sleep(0.5)

        self.is_inference_ready = False
        self.is_inference_running = False
        self.get_chatgpt_text_ready = False
        self.get_chatgpt_text_running = False
        self.get_video_ready = False
        self.get_video_running = False
        self.is_play_audio_running = False
        self.keyword_reset = True

    def play_audio(self):
        if self.label:
            if self.label.winfo_ismapped():
                self.label.pack_forget()
        self.is_play_audio_running = True
        t_play = threading.Thread(target=self.play, args=())
        t_play.start()

    def run_inference(self):
        self.is_inference_running = True
        t_inf = threading.Thread(target=self.run_inference_t, args=())
        t_inf.start()

    def get_chatgpt_text(self, text):
        self.get_chatgpt_text_running = True
        t_inf = threading.Thread(target=self.get_chatgpt_text_t, args=(text,))
        t_inf.start()

    def get_video(self):
        self.get_video_running = True
        t_inf = threading.Thread(target=self.get_video_t, args=())
        t_inf.start()        
   
    def get_video_t(self):
        link = self.get_video_link(self.keyword)
        self.download_video(link)
        self.get_video_ready = True
        self.get_video_running = False

    def get_video_link(self, query):

        # if keyword gives no results, reset keyword

        while True:
            try:
                videosSearch = None
                while not videosSearch:
                    videosSearch = VideosSearch(query, limit = 20) #not enough?
                    time.sleep(0.5)       

                # check durations and find best
                
                results = videosSearch.result()
                filtered = []
                for x in results['result']:
                    print("x var")
                    print(x)
                    if not "none" in x['duration']:
                        d = x['duration'].split(':')[0]
                        if int(d) <= 10 and int(d) >= 3:    
                            filtered.append(x)

                r = random.randint(0,len(filtered)-1)
            except:
                print("Video search failed, trying again with different keyword!")
                i = len(organisms)
                query = organisms[random.randint(0, i-1)]
                time.sleep(0.5)
            else:
                break

        print("video search done")

        #get video title
        self.video_title = filtered[r]['title']

        return filtered[r]['link'] 
        #return results['result'][0]['link']

    def download_video(self, link):
        if os.path.exists("prep_video.mp4"):
            os.remove('prep_video.mp4')
        # os.system("yt-dlp {} -f 137 -o main_video.%(ext)s".format(link))
        # os.system("yt-dlp {} -f 136 -o prep_video.%(ext)s".format(link))
        os.system("yt-dlp {} -f bv*[ext=mp4] -o prep_video.%(ext)s".format(link))

    def get_chatgpt_text_t(self, text):

        #check timer for rate limit
        while (time.time() - self.start_time) < 240:
            print("Waiting for rate limit timer")
            print((time.time() - self.start_time))
            time.sleep(5)
        self.start_timer()

        session_token = 'eyJhbGciOiJkaXIiLCJlbmMiOiJBMjU2R0NNIn0..YXVZD6ZXGxJvJGx8.jj-7qK49XFPZK1Q29ijjTateXfkv1TAW69h70f37Bw_9_qpTtmWcCZJJNwWQE-1C4lXZ8SwnGM-N93yaZv5nw_KJ2VTk2ImeAczmg7kQxulik_49dtJe_0L8LqHe37zbA8LCw9DsOcBvr72a82Sr8Zp_hJXcsSWwP5wi-QZet4ubHK9DDfgskRSJeu2okexFJPgm4Fcus717gtAeCpWyXURLtlZ9lk7HAiF16f0O7oNFJVpC2IGZcf0B9tc3yfJRvvrKJOcJmdE5dwGuIV9ldoaey6Vd9buUYwYAxHUEmxQY5FTBhYsRY4qnzG-sIrpPFO2HI6yFLp9I8At-34qn2szEdfMHiBhoOANmic1qOgBepqefDvw6N9m-1NM12zlcSSuLD86tePL_9H_iLkEjMfDVGl_KDjRKaBkYlb_N8Qk-8ri1VqG6PuJnxUtMhe1THQC9V_5cpY-NkFr2-JrZ5nIR-wSFO4nAPbZM9cDSiMjHj95KjkIwrm7mcy2GI7sXfXN1b-tOoEx5highi8J_f0cfd9LYahbAy465ko-iB4YOg_ImMHn4QtCQI_5ZbCNU-w6P-FtbBEqF2En_4X2uF8BfnRlidwjoq2sGmvm1HrZ9iGs5dzE7G9HgvMD5cN-T7-HpdnjSiX1YVOcxobdjcwF0LzxZ5_qYorW4lGAmhUVBmvlcSEEiI4avT4Cioy5yWZMM0y1rSjyx-kEjfN6br2JA6yz7zmQD5_fs67dnt7g6u9q0AzGkZ_WRUp_9D4RsQWFZNds4akCO2qznPfGPtAWaJNKTcryrgDEWPaj1NcP5Sw3GAjNtvtUvBeqm63gYBZwXb8HtN_oZcRRkGuRvxECRpj25NTzj3ZVIerua-38cfIF7U18FTmSr30VzqrKHtycm0xwrrp50fC1r9ln6m34MSSxm7KBybcL3I9C23CvuJxZPjUu7u-0_5dKa6fX9Qdgb8ojmZOYBzDQvjvyGGD4rVmlPp60dpug_mWRzlwawHt9_cEO0ZWba_3nAq15H19LIjn3V17E44HWiww3gpDjB6gOUr6TMRnVMeGoEyQqbrNULkUk4OsHFAlf1bFCyto5hHSzWhUDonQR3pGMFaurgt1Vp22NP08gcRRMlOU9Oe11Q7ZjKiuWFbKUtXGPmb2NHVP_cgIPLNrAweoSI_6i7qNx-BZMSA9c2SFLDC8tSfAVZ0anIGWdYtjIL5SOWexGcMsXWeANjpyyWO3LdO5uEVK2UtMOSgJznsoNA0mCtMZwXTzaB8zuB4-SzpZYK-fqTNLL7_ZIo8XyI8kuzlDuUHwBsV_XtGVxfWKeNJdJI0lhYxcyF1kS96EXURHWlMzncG7gVQ83wycPIy9ZrihUZIllZtjz40oZAkTzRVMinT5W2p9oGTeFlrqsbUdUW5FDULahN8n4E9ML0gGFTicAfAGQextJWTHruG3Bn-87DG_0YkWA6Pz_nkV0K8-uPG7Osw4P8jTRbrRNZAb9qEZUJqYneQksPKFxJV4SlFnJ9IIyNuAasgV9mKVheeBrAy41FJhKWNFpIv6bgAMKMkRG1Ndl0ExusKwlhtGMpA9zqKcQqaPW53-Euh7qR5aeFUVc6BddGHa0Kpn7VVvYOTFdxH_WfJwwjbBWXy5BjOU2g6O_yG8WE7VC-FDYcZaWAnFuNSR76EoVOjGsiV3xxdpLX_Z1qfB22w723NCNcmMxo9Vw3W0PK5q57JAqC7B1inXrln45DsmK2ZF5E0meR0bmiqnIdYtwmjg7Kt-naJUmbfQU8o-zw03QMUqH62n_8fn0pKKCYPvmGpi0vZRQOl8-jOfM-CnO_YJfXQQMqLuwtavjDXvQ5HBhZgAYjsz4s3RbidBg3mTu-u1zqlX5Q0i3G79LWYwCCgVZ3CFuiRjNHsa3lwDhYhZ9njY_u7tkER_J1I2z9kjWpZVx9PVI3ZqrOZKQkRjDjymG6s1eFdDpEQM-krwmP_FLvhyfDK4QrrxwyJagAFs8uJxLgEXMba7gwOOz5iuS5aXUfypWkvMdTOwrj3dWFgajQ-BdbAT7Qpt0XWFwdXstqkkGCmM4JHRkhAM3FY_wP3LBTinXb0CkMP8odMlIVJtKmALGBVX03xjCivQHEXA78DOFO15wZc2A0M3NMsrYmHPE4KvwqjFmcrV7iv5IPsRRwFxZ6WbA0tkAOFnMymwrh9dcYmd8sYqd465zEj0eCyumBbVOtMPvvU_NVkIC44GXnOa9mq0HOQ6FW7cXDGvhKAz6zQuOVeSgcTMKEEd18fYk-k0K9guov1BcdxjqfkKSnl2KZr0rRuKyyz0HDQ3LIkErMos2r-u1-LRRxB76F_SvvcO7YrwOizD-5_l7JOiSgjSYoavIZJOAxyyZ9dZerIxupOW9I0e49cSOTq2q3HsDw9dpad2QjT8Gl6Rw2431Ll4-yAxyOOTpo8EHC6XPgoDGakH_RTAGxIIKSYCZAKQeNRxWQhNkhzaFfGjkqnqNwQpPhx5jKSGH63uha-XxUDYc6-yKxBhVOQfRgpVNmsokqTuR1JXFg59FTAs1hpzgKXp_RDkximSIqS55w68E7YQ2h-Hk6_7Kyi5JdR8hfoE5GWEoCGOWiuH-zFlBqmnXcLHDscrNuWxKwjpzEc0F0gF0_0ZXZIQSVY9Lq0jimiDExXjho-ZudZX2SjltX98AWioNotbLPjWN-MuqwZfAMa5lU1jQUoMDB.StA2mwwRYI8Trwy0i580-Q' 
        
        
        api = ChatGPT(session_token)
        
        while True:
            try:
                print("Sending GPT message")
                t = api.send_message(text)
            except:
                api.__del__()
                api = ChatGPT(session_token)
                # page crapped out, time to reload
            else:
                break
            time.sleep(10)

        #api.reset_conversation()
        self.chatgpt_message = t['message']
        # t = "Good day to all, it's David Attenborough here. Today, I would like to talk to you about one of the most fascinating creatures in the animal kingdom - the Wombat. Wombats are short-legged, sturdy and heavy creatures with a reputation for being slow-moving and docile."

        #self.chatgpt_message = t

        api.__del__()
        self.get_chatgpt_text_ready = True
        self.get_chatgpt_text_running = False

    def play_main_video(self):
        vlcmedia = self.vlc_obj.media_new("main_video.mp4")
        self.vlcplayer.set_media(vlcmedia)
        self.vlcplayer.set_hwnd(video_frame.winfo_id())#tkinter label or frame

        self.vlcplayer.audio_set_mute(True)
        self.vlcplayer.play()

        self.title_label = tk.Label(root, text="{}, {}".format(self.keyword, self.video_title), font=("TkDefaultFont", 15), bg='black', fg='white')
        self.title_label.pack(side="bottom", anchor="sw")
        video_title_timer = time.time()

        playing = set([1,2,3,4])
        play = True
        while play:
            root.update_idletasks()
            root.update()
            if time.time() - video_title_timer >= 10:
                if self.title_label.winfo_ismapped():
                    self.title_label.pack_forget()
            time.sleep(1)
            if self.is_audio_playing == False:
                play = False
            state = self.vlcplayer.get_state()
            if state in playing:
                continue
            else:
                play = False
        print("playing done")
        self.label = tk.Label(root, text="Generating next episode", font=("TkDefaultFont", 20), width=20, bg='black', fg='white')
        self.label.pack(side="bottom", anchor="se")

    
    def play_bumper_video(self):
        pass

    def run_inference_t(self):
            #Split text 
            text = self.chatgpt_message
            text = text.strip('\n')
            text = text.strip('"')
            text = text.strip('”')
            text = text.strip('“')            
            text_list = re.split(r'(?<=[\.\!\?])\s*', text)
            text_list = [i for i in text_list if i]
            print(text_list)
            wav_list = []

            for i, t in enumerate(text_list):
                os.makedirs("wav_out", exist_ok=True)
                wav_file = "wav_out/out{}.wav".format(i)        
                sequence = np.array(text_to_sequence(t, ['english_cleaners']))[None, :]
                sequence = torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()
                mel_outputs, mel_outputs_postnet, _, alignments = self.taco_model.inference(sequence)    
                
                print("Using waveglow model")
                for k in self.waveglow_model.convinv:
                    k.float()
                self.denoiser = Denoiser(self.waveglow_model) 
                with torch.no_grad():
                    audio = self.waveglow_model.infer(mel_outputs_postnet, sigma=1)
                audio_denoised = self.denoiser(audio, strength=0.02)[:, 0]
                audioout = audio_denoised[0].data.cpu().numpy()
                #audioout = audio[0].data.cpu().numpy()
                audioout32 = np.float32(audioout)
                sf.write(wav_file, audioout32, 22050)
                wav_list.append(wav_file)
                time.sleep(0.5)
            self.wav_list = wav_list            
            self.is_inference_ready = True
            self.is_inference_running = False
            
# Create tkinter window
root = tk.Tk()
root.title("VLC player in Tkinter")
root.geometry("1280x720")

video_frame = tk.Frame(root, bg="black")
video_frame.pack(fill=tk.BOTH, expand=True)

# # start background music

# background_music = vlc.MediaPlayer("classical.mp3")
# background_music.audio_set_volume(70)
# background_music.play()

control = Control()
control.run()
root.mainloop()

