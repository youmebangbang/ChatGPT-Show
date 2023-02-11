from pyChatGPT import ChatGPT
from youtubesearchpython import VideosSearch
import pafy 
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

organisms = ["Quercus alba", "Pinus sylvestris", "Acer rubrum", "Populus tremuloides", "Betula papyrifera", "Picea abies", "Abies balsamea", "Salix babylonica", "Populus balsamifera", "Alnus incana", "Fraxinus americana", "Castanea sativa", "Juglans nigra", "Liriodendron tulipifera", "Tsuga canadensis", "Ginkgo biloba", "Carya ovata", "Sassafras albidum", "Ulmus americana", "Fagus grandifolia", "Cercis canadensis", "Rhododendron maximum", "Sorbus americana", "Carpinus caroliniana", "Prunus serotina", "Crataegus crus-galli", "Ostrya virginiana", "Tilia americana", "Malus pumila", "Hippophae rhamnoides", "Euonymus americanus", "Corylus americana", "Amelanchier arborea", "Pyrus communis", "Acer saccharum", "Rhamnus cathartica", "Zelkova serrata", "Diospyros virginiana", "Lonicera japonica", "Sambucus canadensis", "Viburnum dentatum", "Viburnum opulus", "Vitis vinifera", "Vaccinium corymbosum", "Arbutus unedo", "Cornus sericea", "Cornus stolonifera", "Ilex opaca", "Quercus robur", "Aesculus hippocastanum", "Liriodendron tulipifera", "Cedrus deodara", "Cedrus atlantica", "Pinus wallichiana", "Pinus ponderosa", "Pinus cembra", "Pinus mugo", "Pinus rigida", "Pinus sylvestris var. hamata", "Pinus sylvestris var. mughus", "Pinus sylvestris var. mongolica", "Pinus banksiana", "Pinus resinosa", "Pinus contorta", "Pinus parviflora", "Pinus nigra", "Pinus canariensis", "Pinus halepensis", "Pinus pinaster", "Pinus densiflora", "Pinus pinea", "Pinus peuce", "Pinus jeffreyi", "Pinus longaeva", "Pinus coulteri", "Pinus torreyana", "Pinus radiata", "Pinus leiophylla", "Pinus monophylla", "Pinus strobus", "Pinus ponderosa var. scopulorum", "Pinus strobus var. chiapasensis", "Pinus strobus var. strobus", "Pinus strobus var. ecuadoriensis", "Pinus strobus var. neostrobus", "Pinus strobus var. pendula", "Pinus strobus var. pseudostrobus", "Pinus strobus var. strobiformis", "Pinus strobus var. virginiana", "Pinus virginiana", "Pinus echinata", "Pinus clausa", "Pinus palustris", "Pinus serotina", "Pinus virginiana var. virginiana", "Homo sapiens", "Canis lupus", "Felis catus", "Panthera leo", "Equus caballus", "Bos taurus", "Ovis aries", "Cervus elaphus", "Sus scrofa", "African Elephant", "Lion", "Giraffe", "Crocodile", "Hippopotamus", "Rhino", "Gorilla", "Cheetah", "Baboon", "Buffalo", "Kudu", "Zebra", "Leopard", "Hyena", "Wildebeest", "Impala", "Antelope", "Hartebeest", "Eland", "Warthog", "Ostrich", "Chimpanzee", "Jaguar", "Monkey", "Nyala", "Ocelot", "Civet", "Bushbaby", "Baboon Spider", "Bison", "Chacma Baboon", "Common Duiker", "Cape Eland", "Giant Forest Hog", "Sable Antelope", "Buffalo Weaver", "Black Rhinoceros", "Red Duiker", "Gazelle", "Vervet Monkey", "African Wild Dog", "Lioness", "Common Waterbuck", "Plains Zebra", "Klipspringer", "Yellow-Backed Duiker", "Reedbuck", "Spotted Hyena", "African Elephant Calf", "African Buffalo Calf", "African Elephant Family", "Leopard Cub", "Lion Cub", "Giraffe Calf", "Lioness and Cubs", "Lion Pride", "Chimpanzee Family", "Gorilla Family", "Rhino Calf", "Hippo Calf", "Baboon Troop", "African Wild Dog Pack", "Giraffe Herd", "Elephant Herd", "Zebra Herd", "Antelope Herd", "Lioness Hunting", "Gorilla Eating", "Rhino Drinking", "Hippo Wallowing", "Chimpanzee Playing", "Lion Roaring", "Giraffe Stretching", "Cheetah Running", "African Elephant Trunk", "Lion Mane", "Giraffe Neck", "Crocodile Mouth", "Hippopotamus Teeth", "Rhino Horn", "Gorilla Chest", "Cheetah Spot", "Baboon Bottom", "Buffalo ", "Kudu Horns", "Zebra Stripes", "Leopard Print", "Hyena Laugh", "Wildebeest Migration", "Impala Jump", "Antelope Grazing", "Hartebeest Charge", "Eland Charge", "Warthog Tusk", "Ostrich Feather", "Chimpanzee Smile", "Jaguar Roar", "Monkey Swing", "Nyala Horns", "Ocelot Whiskers", "Civet Tail", "Bushbaby Eyes", "Baboon Spider Web", "Bison ", "Chacma Baboon Face", "Common Duiker Horns", "Cape Eland Antlers", "Giant Forest Hog Tusk", "Sable Antelope Horns", "Buffalo Weaver Nest", "Black Rhinoceros Horn", "Red Duiker Horns", "Gazelle Hooves", "Vervet Monkey Call", "Asian Elephant", "Bengal Tiger", "Indian Rhinoceros", "Himalayan Bear", "Gaur", "Wild Water Buffalo", "Snow Leopard", "Sambar Deer", "Kouprey", "Hippopotamus", "Asiatic Lion", "Goral", "Nilgai", "Civet", "Sloth Bear", "Banteng", "Indian Mongoose", "Barbary Macaque", "Golden Langur", "Clouded Leopard", "Black-tailed Deer", "Indian Grey Mongoose", "Lar Gibbon", "Indian Pangolin", "Malabar Giant Squirrel", "Grizzled Giant Squirrel", "Bharal", "Muntjac", "Marbled Cat", "Small Indian Civet", "Striped Hyena", "Jungle Cat", "Serow", "Leopard Cat", "Asian Black Bear", "Brown Palm Civet", "Large-tailed Nightjar", "Yellow-throated Marten", "Pygmy Hog", "Indian Wolf", "Red Panda", "Bengal Fox", "Greater One-horned Rhinoceros", "Lesser One-horned Rhinoceros", "Nepal Grey Langur", "Hog Deer", "Spotted Linsang", "Chestnut-bellied Sandgrouse", "Blyth's Tragopan", "Chital", "Sri Lankan Leopard", "Sri Lankan Sloth Bear", "Sri Lankan Elephant", "Golden Jackal", "Bengal Slow Loris", "Plains Zebra", "Hornbill", "Pangolin", "Wild Boar", "Indian Peafowl", "Pelican", "Brahminy Kite", "Osprey", "Ibis", "Eagle", "Stork", "Heron", "Kingfisher", "Pheasant", "Parakeet", "Flamingo", "Hornbilled Duck", "Sparrow", "Robin", "Pigeon", "Dove", "Peacock", "Crow", "Owl", "Falcon", "Vulture", "Kite", "Swan", "Duck", "Herring Gull", "Cormorant", "Seagull", "Tern", "Osprey", "Kestrel", "Hawk", "Eagle Owl", "Eagle-owl", "Long-tailed Shrike", "Magpie", "Cuckoo", "Drongo", "Thrush", "Starling", "Lark", "Sunbird", "Bee-eater", "Kingfisher", "Green Pigeon", "Parrot", "Woodpecker", "Nightjar", "Swift", "Wagtail", "Finch", "Pelican", "Bustard", "Shorebird", "Sandpiper", "Plover", "Lapwing", "Stilt", "Dotterel", "Teal", "Pintail", "Pochard", "Mallard", "Gadwall", "Wigeon", "Shoveler", "Pochard", "Goose", "Swan", "Crane", "Stork", "Heron", "Ibis", "Egret", "Bittern", "Spoonbill", "Elephas maximus", "Bos gaurus", "Bos javanicus", "Bos frontalis", "Bubalus bubalis", "Bubalus arnee", "Cervus unicolor", "Cervus nippon", "Sus scrofa", "Sus barbatus", "Suidae sp.", "Capricornis sumatraensis", "Hippopotamus amphibius", "Hippopotamus amphibius variegatus", "Rhinoceros unicornis", "Rhinoceros sondaicus", "Dicerorhinus sumatrensis", "Tursiops aduncus", "Orcinus orca", "Ganges River Dolphin", "Lagenodelphis hosei", "Platanista gangetica", "Platanista minor", "Vicugna pacos", "Lama glama", "Lama guanicoe", "Alpaca Pacos", "Ovis aries", "Capra hircus", "Nemorhaedus goral", "Nemorhaedus sumatraensis", "Serow Nemorhaedus", "Thinhorn Sheep", "Ovis ammon", "Antilope cervicapra", "Antilope saiga", "Gazella subgutturosa", "Gazella bennettii", "Gazella gazella", "Panthera tigris", "Panthera pardus", "Panthera onca", "Panthera leo", "Jaguar Panthera", "Leopardus pardalis", "Leopardus wiedii", "Snow Leopard", "Leopardus uncia", "Felis bengalensis", "Felis silvestris", "Felis manul", "Felis margarita", "Canis lupus", "Canis aureus", "Canis rufus", "Ursus arctos", "Ursus thibetanus", "Asiatic Black Bear", "Ursus malayanus", "Ursus maritimus", "Pinnipedia", "Otariidae", "Otaria flavescens", "Zalophus californianus", "Eumetopias jubatus", "Phocidae", "Erignathus barbatus", "Leptonychotes weddellii", "Mirounga angustirostris", "Halichoerus grypus", "Lobodon carcinophaga", "Hydrurga leptonyx", "Crabeater Seal", "Leopard Seal", "Ommatophoca rossii", "Ross Seal", "Phocarctos hookeri", "Arctocephalus forsteri", "Arctocephalus galapagoensis", "Arctocephalus philippii", "Arctocephalus townsendi", "Lontra canadensis", "Lontra felina", "Mustela lutreola", "Mustela erminea", "Mustela nivalis", "Mustela putorius", "Martes zibellina", "Martes americana", "Viverridae", "Viverra zibetha", "Viverra civettina", "Viverra tangalunga", "Herpestidae", "Herpestes javanicus", "Herpestes smithii", "Herpestes ichneumon", "Cynictis penicillata", "Abelmoschus moschatus", "Acorus calamus", "Adiantum capillus-veneris", "Aegle marmelos", "Aesculus turbinata", "Alocasia macrorrhiza", "Alpinia galanga", "Alpinia oxyphylla", "Amla", "Anacardium occidentale", "Anemarrhena asphodeloides", "Angelica dahurica", "Antennaria dioica", "Artemisia annua", "Artemisia apiacea", "Artemisia capillaris", "Artemisia japonica", "Artemisia vulgaris", "Asarum sieboldii", "Asclepias syriaca", "Asparagus racemosus", "Atractylodes lancea", "Bambusa arundinacea", "Bambusa tulda", "Berberis aquifolium", "Bergenia crassifolia", "Betula platyphylla", "Bischofia javanica", "Bixa orellana", "Borago officinalis", "Boswellia carterii", "Brassica rapa", "Broussonetia papyrifera", "Camellia sinensis", "Canna indica", "Capsicum annuum", "Carica papaya", "Carum carvi", "Carya illinoinensis", "Carya ovata", "Carya sinensis", "Castanea mollissima", "Cedrella sinensis", "Celastrus orbiculatus", "Centella asiatica", "Cercis chinensis", "Chamaemelum nobile", "Chrysanthemum coronarium", "Cinnamomum camphora", "Cinnamomum verum", "Cistanche deserticola", "Cistus incanus", "Citrullus lanatus", "Clinopodium acinos", "Clitoria ternatea", "Cocos nucifera", "Commiphora myrrha", "Coptis chinensis", "Cornus officinalis", "Crataegus pinnatifada", "Cryptomeria japonica", "Cucurbita pepo", "Cupressus funebris", "Cupressus sempervirens", "Cydonia oblonga", "Cymbopogon citratus", "Cyperus rotundus", "Daucus carota", "Dendrocalamus giganteus", "Dendrobium candidum", "Dendrobium fimbriatum", "Dendrobium nobile", "Dendrobium ochreatum", "Desmodium gangeticum", "Diospyros kaki", "Dolichos lablab", "Echinacea purpurea", "Elaeagnus angustifolia", "Elaeis guineensis", "Eleutherococcus senticosus", "Emblica officinalis", "Erythrina variegata", "Eucalyptus globulus", "Eucommia ulmoides", "Ficus carica", "Ficus religiosa", "Garcinia cambogia", "Glycyrrhiza glabra", "Gossypium hirsutum", "Gynura segetum", "Hedera helix", "Hibiscus sabdariffa", "Hypericum perforatum", "Ilex paraguariensis", "Illicium verum", "Indigofera tinctoria", "Jasminum officinale", "Juglans regia", "Juniperus chinensis", "Kadsura japonica", "Laminaria japonica", "Larix gmelinii", "Lavandula angustifolia", "Lepidium meyenii", "Lindera aggregata", "Lonicera japonica", "Lycium barbarum", "Magnolia kobus", "Magnolia officinalis", "Mangifera indica", "Mentha piperita", "Metasequoia glyptostroboides", "Micromelum pubescens", "Morus alba", "Morus nigra", "Nelumbo nucifera", "Nepeta cataria", "Olea europaea", "Oroxylum indicum", "Panax ginseng", "Panicum miliaceum", "Papaver somniferum", "Paradisea liliastrum", "Paeonia lactiflora", "Pelargonium graveolens", "Pelargonium x hortorum", "Perilla frutescens", "Picea abies", "Pinus armandii", "Pinus densiflora", "Pinus massoniana", "Pinus sylvestris", "Pistacia chinensis", "Pistacia vera", "Pleurotus ostreatus", "Podocarpus macrophyllus", "Polygonum multiflorum", "Prunus avium", "Prunus cerasus", "Prunus dulcis", "Prunus persica", "Punica granatum", "Raphanus sativus", "Rhamnus cathartica", "Rhododendron indicum", "Ricinus communis", "Robinia pseudoacacia", "Rosa chinensis", "Rosa rugosa", "Rubus idaeus", "Salix babylonica", "Sambucus nigra", "Sapindus mukorossi", "Sassafras albidum", "Schisandra chinensis", "Scutellaria baicalensis", "Sesamum indicum", "Silybum marianum", "Smilax glabra", "Sophora flavescens", "Sophora japonica", "Soymida febrifuga", "Styrax japonicus", "Swertia japonica", "Syzygium aromaticum", "Taxus chinensis", "Taxus cuspidata", "Thymus vulgaris", "Tilia cordata", "Tinospora cordifolia", "Toona sinensis", "Ulmus parvifolia", "Urtica dioica", "Vaccinium myrtillus", "Vaccinium oxycoccos", "Vaccinium uliginosum", "Vitis vinifera", "Zingiber officinale", "Acanthocalycium klimpelianum", "Acerola", "Alchornea castaneifolia", "Amaranthus hypochondriacus", "Annona cherimola", "Anthurium andraeanum", "Aralia elata", "Ardisia elliptica", "Arrabidaea chica", "Arum maculatum", "Aspidosperma cylindrocarpon", "Astrocaryum murumuru", "Attalea phalerata", "Bactris gasipaes", "Bauhinia forficata", "Berberis darwinii", "Bixa orellana", "Bourreria ovata", "Brachychiton populneus", "Bromelia karatas", "Brunfelsia grandiflora", "Cabralea canjerana", "Cajanus cajan", "Calathea lutea", "Calathea zebrina", "Calotropis gigantea", "Calyptranthes concinna", "Calyptranthes pauciflora", "Calyptranthes spinosa", "Calyptrogyne ghiesbreghtiana", "Calyxochoma spectabilis", "Camellia sinensis", "Campomanesia xanthocarpa", "Cananga odorata", "Capsicum annuum", "Carica papaya", "Caryocar villosum", "Cecropia obtusifolia", "Cestrum aurantiacum", "Chrysophyllum cainito", "Cissus verticillata", "Clausena anisata", "Clusia rosea", "Coccoloba uvifera", "Coffea arabica", "Columnea xanthiifolia", "Conostegia xalapensis", "Copernicia alba", "Copernicia baileyana", "Copernicia cerifera", "Copernicia prunifera", "Copernicia tectorum", "Cordia goeldiana", "Cordia alliodora", "Costus spicatus", "Couma macrocarpa", "Crataegus monogyna", "Cryptocarya alba", "Cupania americana", "Cupressus sempervirens", "Cyathea medullaris", "Cyathea lepifera", "Dendropanax cuneatus", "Dendropanax morototoni", "Dendropanax trifidus", "Desmanthus virgatus", "Dimorphandra mollis", "Dimorphotheca sinuata", "Dioscorea alata", "Drymis winteri", "Duschekia fruticosa", "Eichhornia crassipes", "Elettaria cardamomum", "Embelia ribes", "Enterolobium cyclocarpum", "Eryngium foetidum", "Erythrina crista-galli", "Eschweilera coriacea", "Eucalyptus citriodora", "Eucalyptus grandis", "Eucalyptus marginata", "Eucalyptus regnans", "Euphorbia heterophylla", "Euterpe edulis", "Ficus benjamina", "Ficus elastica", "Ficus retusa", "Garcinia dulcis", "Ginger", "Gmelina arborea", "Guazuma ulmifolia", "Heteropterys auriculata", "Hibiscus rosa-sinensis", "Hydrocotyle ranunculoides", "Hymenaea courbaril", "Inga edulis", "Ipomoea batatas", "Jacaranda mimosifolia", "Juglans regia", "Juniperus communis", "Lantana camara", "Litchi chinensis", "Lonchocarpus nitidus", "Lucuma salicifolia", "Malpighia glabra", "Mangifera indica", "Manihot esculenta", "Matayba elaeagnoides", "Mauritia flexuosa", "Melaleuca leucadendron", "Melastoma malabathricum", "Melastomataceae", "Mitracarpus villosus", "Myrciaria dubia", "Myrtaceae", "Nectandra rodiaei", "Nectandra sanguinea", "Ocimum basilicum", "Orchidaceae", "Oreopanax daphnifolius", "Oreopanax xalapensis", "Oryza sativa", "Oxandra lanceolata", "Pachira aquatica", "Paederia scandens", "Pancratium maritimum", "Peperomia pellucida", "Persea americana", "Petiveria alliacea", "Phaseolus lunatus", "Philodendron bipinnatifidum", "Philodendron scandens", "Physalis peruviana", "Picea abies", "Pimenta dioica", "Pinus radiata", "Piper auritum", "Piper nigrum", "Pitcairnia heterophylla", "Pleiostachya pruinosa", "Podocarpus nagi", "Polypodiaceae", "Pouteria campechiana", "Pouteria sapota", "Prunus avium", "Psidium guajava", "Pterocarpus rohrii", "Punica granatum", "Rheedia edulis", "Rosa canina", "Rosmarinus officinalis", "Rubus fruticosus", "Sambucus nigra", "Sapindus saponaria", "Sapium glandulosum", "Sassafras albidum", "Scutia buxifolia", "Serenoa repens", "Solanum lycopersicum", "Solanum quitoense", "Solanum tuberosum", "Sterculia apetala", "Swietenia mahagoni", "Symplocos cochinchinensis", "Talauma ovata", "Tecoma stans", "Tithonia diversifolia", "Tocoyena formosa", "Tropaeolum majus", "Ulex europaeus", "Urtica dioica", "Vaccinium meridionale", "Vernonia polyanthes", "Viburnum tinus", "Vochysia guatemalensis", "Wisteria floribunda", "Zea mays", "Amazona aestiva", "Andean Condor", "Vultur gryphus", "Anhima cornuta", "Southern Screamer", "Rhea americana", "Ovis aries", "Lama glama", "Alpaca pacos", "Vicugna pacos", "Ceratotherium simum", "Elephantulus rufescens", "Hydrochoerus hydrochaeris", "Myocastor coypus", "Otaria flavescens", "Callithrix jacchus", "Callicebus moloch", "Saguinus geoffroyi", "Saimiri boliviensis", "Ateles belzebuth", "Lagothrix lagotricha", "Cacajao calvus", "Aotus nancymaae", "Cavia porcellus", "Dasyprocta aguti", "Agouti paca", "Geocapromys brownii", "Tayassu pecari", "Dasypus novemcinctus", "Bradypus variegatus", "Cougar", "Puma concolor", "Jaguar", "Panthera onca", "Ocelot", "Leopardus pardalis", "Oncifelis colocolo", "Felis concolor", "Panthera tigris", "Lion", "Acinonyx jubatus", "Cheetah", "Eira barbara", "Tayassu tajacu", "Sus scrofa", "Wild Boar", "Lama guanicoe", "Guanaco", "Vicugna vicugna", "Llama", "Dama dama", "Fallow Deer", "Odocoileus virginianus", "Alces alces", "Moose", "Cervus elaphus", "Red Deer", "Rangifer tarandus", "Reindeer", "Procyon lotor", "Raccoon", "Ursus americanus", "Black Bear", "Potos flavus", "Kinkajou", "Nasua nasua", "Coati", "Galeopterus variegatus", "Flying Squirrel", "Sciurus carolinensis", "Tamandua tetradactyla", "Anteater", "Conepatus semistriatus", "Skunk", "Myotis velifer", "Big Brown Bat", "Lasionycteris noctivagans", "Silver-haired Bat", "Eptesicus fuscus", "Big Brown Bat", "Lasiurus cinereus", "Hoary Bat", "Phyllostomus hastatus", "Pallas's Long-tongued Bat", "Chiroptera", "Bats", "Eumops perotis", "Hooded Bat", "Glossophaga soricina", "Common Long-tongued Bat", "Didelphis marsupialis", "Common Opossum", "Conepatus chinga", "Andean Skunk", "Conepatus leuconotus", "White-backed Skunk", "Mephitis mephitis", "Striped Skunk", "Spilogale putorius", "Skunk", "Chrysocyon brachyurus", "Maned Wolf", "Canis latrans", "Coyote", "Canis lupus", "Gray Wolf", "Canis simensis", "Lycalopex gymnocercus", "Patagonian Fox", "Pseudalopex culpaeus", "Darwin's Fox", "Lontra canadensis", "North American River Otter", "Lontra longicaudis", "Southern River Otter", "Eira barbara", "Tayassu tajacu", "Sus scrofa", "Wild Boar", "Felis concolor", "Panthera tigris", "Lion", "Acinonyx jubatus", "Cheetah", "Hydrochoerus hydrochaeris", "Myocastor coypus", "Otaria flavescens", "Callithrix jacchus", "Callicebus moloch", "Saguinus geoffroyi", "Saimiri boliviensis", "Ateles belzebuth", "Lagothrix lagotricha", "Cacajao calvus", "Aotus nancymaae", "Cavia porcellus", "Dasyprocta aguti", "Agouti paca", "Geocapromys brownii", "Tayassu pecari", "Dasypus novemcinctus", "Bradypus variegatus", "Vultur gryphus", "Andean Condor", "Cervus elaphus", "Red Deer", "Rangifer tarandus", "Reindeer", "Alces alces", "Moose", "Odocoileus virginianus", "Fallow Deer", "Lama glama", "Alpaca pacos", "Vicugna pacos", "Ceratotherium simum", "Elephantulus rufescens", "Ovis aries", "Rhea americana", "Southern Screamer", "Anhima cornuta", "Amazona aestiva", "Phyllostomus hastatus", "Pallas's Long-tongued Bat", "Chiroptera", "Bats", "Eumops perotis", "Hooded Bat", "Glossophaga soricina", "Common Long-tongued Bat", "Lasionycteris noctivagans", "Silver-haired Bat", "Eptesicus fuscus", "Big Brown Bat", "Lasiurus cinereus", "Hoary Bat", "Myotis velifer", "Big Brown Bat", "Mephitis mephitis", "Striped Skunk", "Spilogale putorius", "Skunk", "Conepatus leuconotus", "White-backed Skunk", "Conepatus chinga", "Andean Skunk", "Didelphis marsupialis", "Common Opossum", "Nasua nasua", "Coati", "Potos flavus", "Kinkajou", "Ursus americanus", "Black Bear", "Procyon lotor", "Raccoon", "Tamandua tetradactyla", "Anteater", "Sciurus carolinensis", "Flying Squirrel", "Galeopterus variegatus", "Anteater", "Cougar", "Puma concolor", "Jaguar", "Panthera onca", "Ocelot", "Leopardus pardalis", "Oncifelis colocolo", "Canis lupus", "Gray Wolf", "Canis latrans", "African Elephant", "Lion", "Giraffe", "Zebra", "Hippopotamus", "Buffalo", "Rhinoceros", "Cheetah", "Leopard", "Gorilla", "Chimpanzee", "Baboon", "Warthog", "Antelope", "Hyena", "Kudu", "Ostrich", "Eland", "Impala", "Klipspringer", "Springbok", "Gazelle", "Wildebeest", "Black Wildebeest", "Blue Wildebeest", "Roan Antelope", "Sable Antelope", "Hartebeest", "Bontebok", "Red Hartebeest", "Tsessebe", "Waterbuck", "Nyala", "Bushbuck", "Bushpig", "Common Duiker", "Yellow-backed Duiker", "Banded Mongoose", "Egyptian Mongoose", "Meerkat", "Giant Forest Hog", "Bushbaby", "Aardvark", "Pangolin", "Squirrel", "Bats", "Lemur", "Chacma Baboon", "Mandrill", "Vervet Monkey", "Patas Monkey", "De Brazza's Monkey", "Baboon Spider", "Banded Gerbil", "Banded Mongoose", "Black-tailed Mongoose", "Cane Rat", "Civet", "Genet", "Honey Badger", "Jackal", "Lamandau", "Liberian Mongoose", "Pangolin", "Ratel", "African Wild Dog", "Black-backed Jackal", "Golden Jackal", "Side-striped Jackal", "Bat-eared Fox", "Bushy-tailed Mongoose", "Egyptian Mongoose", "Gambian Mongoose", "Slender Mongoose", "Small Grey Mongoose", "Weasel", "Yellow Mongoose", "African Civet", "African Giant Rat", "Cape Grey Mongoose", "Four-toed Elephant Shrew", "Namaqua Dwarf Shrew", "Lesser Egyptian Shrew", "Red-tailed Shrew", "Southern Lesser Elephant Shrew", "Southern Tree Shrew", "Tropical House Shrew", "Vlei Rat", "Wandering Shrew", "Yellow-bellied Shrew Mole", "Bush Duiker", "Bathyergid Mole Rat", "Cape Hyrax", "Desert Warthog", "Dwarf Mongoose", "Ethiopian Dwarf Mongoose", "Giant Rats", "Helmeted Guineafowl", "Klipspringer", "Lemming", "Naked Mole Rat", "Prairie Dog", "Rainbow Squirre", "Rock Hyrax", "Rock Squirrel", "Southern Tree Squirrel", "Spiny Mouse", "Suni", "Thick-tailed Bushbaby", "Three-striped Grass Mouse", "Vole", "Wombat", "Angolan Colugo", "Aye-aye", "Banded Lemming", "Bathyergid Mole Rat", "Black-tailed Prairie Dog", "Cape Hare", "Cape Mole-rat", "Cape Porcupine", "Cape Grey Mongoose", "Chestnut Jird", "Cinereus Shrew", "Common Musk Shrew", "Desert Shrew", "Dassie Rat", "Ethiopian Shrew", "Four-toed Elephant Shrew", "Harvest Mouse", "Highveld Gerbil", "House Mouse", "Kalahari Pangolin", "Luzon Shrew", "Malagasy Shrew", "Namaqua Shrew", "African Pygmy Mouse", "Red Veldrat", "Round-tailed Musk Shrew", "Saharan Gerbil", "Southern Lesser Bushbaby", "Striped Ground Squirrel", "Thick-tailed Gerbil", "Thick-tailed Pygmy Squirrel", "Tree Squirrel", "Tufted-tailed Rat", "West African Shrew", "Yellow Mole Rat", "Zebra Mouse", "African Giant Squirrel", "African Hare", "African Mole-rat", "Black-tailed Tree Rat", "Bushveld Gerbil", "Cape Jumping Rat", "Cape Rat", "Crowned Pangolin", "DeBrazza's Guenon", "Egyptian Gerbil", "Ethiopian Bushbaby", "Ethiopian Highland Squirrel", "Flat-headed Gerbil", "Golden Mole", "Giant Golden Mole", "Giant Pangolin", "Grave's Mole Rat", "Ground Pangolin", "Harvest Gerbil", "Highveld Mole Rat", "House Rat", "Kalahari Gerbil", "Kangaroo Rat", "Kenyan Shrew", "Lemming", "Lodgepole Chipmunk", "Malagasy Giant Jumping Rat", "Mediterranean Hare", "Mole Rat", "Northern Tree Shrew", "Red Squirrel", "Round-tailed Ground Squirrel", "Saharan Striped Polecat", "Savanna Shrew", "Southern African Ground Squirrel", "Southern Red-tailed Squirrel", "Striped Squirrel", "Tree Rat", "Tufted-tailed Ground Squirrel", "Umbra Mole Rat", "Vole", "Wandering Shrew", "Yellow-spotted Bushbaby", "African Pangolin", "Black Rat", "Brown Hare", "Bushbaby", "Cape Gerbil", "Cape Ground Squirrel", "Cape Mole Rat", "Cape Shrew", "Cape Squirrel", "Common Squirrel", "Desert Gerbil", "Ethiopian Ground Squirrel", "Ethiopian Rock Squirrel", "Flying Squirrel", "Giant Jumping Rat", "Giant Pouched Rat", "Golden Hamster", "Ground Squirrel", "Harvest Mole Rat", "Highveld Squirrel", "House Gerbil", "Jumping Rat", "Kalahari Ground Squirrel", "Lemming", "Mole Rat", "Pygmy Mole Rat", "Red-tailed Squirrel", "Round-tailed Squirrel", "Saharan Gerbil", "Saharan Jerboa", "Saharan Pygmy Squirrel", "Savanna Squirrel", "Southern Tree Squirrel", "Tree Gerbil", "Yellow Gerbil", "African Baobab Tree", "Aloe Vera", "Anacardiaceae", "Ancylobotrys", "Andropogon", "Angolensis", "Anthurium", "Aralia", "Aristolochia", "Arrowroot", "Artabotrys", "Artocarpus", "Asparagus Fern", "Aspidosperma", "Asystasia", "Ataenidia", "Baker's Alchemilla", "Banana", "Baphia", "Barleria", "Bauhinia", "Begonia", "Beloperone", "Berchemia", "Bergenia", "Berlinia", "Betula", "Bignoniaceae", "Bixa", "Black Pepper", "Boehmeria", "Bojeria", "Bombax", "Borassus", "Brachylaena", "Brachystegia", "Brassia", "Bridelia", "Brownea", "Buddleja", "Butia", "Caladium", "Callistemon", "Calycanthus", "Camellia", "Cannabis", "Canthium", "Capsicum", "Carica", "Cedrela", "Celtis", "Ceratonia", "Cestrum", "Chamomilla", "Chionanthus", "Citrus", "Clerodendrum", "Clivia", "Clusia", "Coffea", "Colocasia", "Commiphora", "Conyza", "Copaifera", "Cordia", "Coriaria", "Coronilla", "Cortaderia", "Crassula", "Crotalaria", "Cryptostegia", "Cucumis", "Curtisia", "Cyathea", "Cymbopogon", "Cynoglossum", "Cyperus", "Dalbergia", "Datura", "Dendrobium", "Dichapetalum", "Dioscorea", "Dracaena", "Drypetes", "Echium", "Elaeis", "Elephantopus", "Encephalartos", "Ensete", "Eryngium", "Eucalyptus", "Euphorbia", "Ficus", "Firmiana", "Flacourtia", "Fragaria", "Gardenia", "Gaultheria", "Gazania", "Gelonium", "Girardinia", "Glycine", "Gossypium", "Grewia", "Grumilea", "Guarea", "Haemanthus", "Haplocarpha", "Harungana", "Hedera", "Helianthus", "Hibiscus", "Hippeastrum", "Holarrhena", "Hydrangea", "Hypericum", "Hypoxis", "Impatiens", "Indigofera", "Inula", "Ipomoea", "Iresine", "Jacaranda", "Jasminum", "Jatropha", "Kigelia", "Kosteletzkya", "Kunzea", "Lactuca", "Lamium", "Lantana", "Lavandula", "Lecomtella", "Lemon", "Lepidagathis", "Leucas", "Leucospermum", "Leycesteria", "Lippia", "Litsea", "Lobelia", "Lonicera", "Loranthus", "Ludwigia", "Luffa", "Lunaria", "Lycopodium", "Macadamia", "Macaranga", "Macrolobium", "Maerua", "Mahonia", "Malva", "Mangifera", "Manilkara", "Manungu", "Maranta", "Marsdenia", "Medicago", "Melia", "Melianthus", "Melocanna", "Melolontha", "Mentha", "Michelia", "Microglossa", "Mikania", "Mimusops", "Mitragyna", "Mnesithea", "Morinda", "Muehlenbeckia", "Mullein", "Myrtus", "Nerium", "Nymphaea", "Ocimum", "Olea", "Ophiopogon", "Oreganum", "Osteospermum", "Oxalis", "Pachypodium", "Papaver", "Paphiopedilum", "Parabenkemia", "Parkinsonia", "Parsley", "Pavonia", "Peucedanum", "Phlomis", "Phygelius", "Phyla", "Phytolacca", "Plectranthus", "Podocarpus", "Polygala", "Polygonum", "Pouteria", "Prangos", "Prunus", "Pteris", "Pulmonaria", "Punica", "Pycnostachys", "Raphionacme", "Raphus", "Rauvolfia", "Razafimandimbisonia", "Rhamnus", "Rhododendron", "Ricinus", "Rosa", "Rostraria", "Ruellia", "Rutaceae", "Saba", "Salicornia", "Salix", "Santolina", "Sapium", "Sassafras", "Schkuhria", "Schotia", "Schrankia", "Scilla", "Sclerocarya", "Senecio", "Senna", "Serenoa", "Sesamum", "Setaria", "Sida", "Sigesbeckia", "Siphonochilus", "Solanum", "Sphagneticola", "Spiraea", "Stachys", "Stapelia", "Strelitzia", "Stylophorum", "Symphoricarpos", "Symplocos", "Tamarindus", "Tarenna", "Taxodium", "Theobroma", "Thunbergia", "Tithonia", "Tradescantia", "Trifolium", "Tulbaghia", "Turnera", "Ulmus", "Uvaria", "Vaccinium", "Vernonia", "Veronica", "Vicia", "Vigna", "Vitis", "Welwitschia", "Xylopia", "Zantedeschia", "Zinnia", "Asiatic Elephant", "Bengal Tiger", "Indian Rhinoceros", "Gaur", "Sambar", "Chital", "Nilgai", "Muntjac", "Hog Deer", "Barasingha", "Kakapo", "Blackbuck", "Himalayan Tahr", "Blue Whale", "Fin Whale", "Minke Whale", "Sei Whale", "Bryde's Whale", "Humpback Whale", "Gray Whale", "Orca", "Bottlenose Dolphin", "Common Dolphin", "Spotted Dolphin", "Striped Dolphin", "Risso's Dolphin", "Spinner Dolphin", "Humpback Dolphin", "Pacific White-Sided Dolphin", "Indian Peafowl", "Great Indian Hornbill", "Indian Roller", "Common Myna", "Green Bee-Eater", "Chestnut-Headed Bee-Eater", "Blue-tailed Bee-Eater", "Lilac-Breasted Roller", "Collared Falconet", "Peregrine Falcon", "Crested Serpent Eagle", "Spotted Eagle", "Steppe Eagle", "Common Kestrel", "Lanner Falcon", "Saker Falcon", "Gyr Falcon", "Red-Necked Falcon", "Himalayan Snowcock", "Himalayan Monal", "Himalayan Partridge", "Gray Peacock Pheasant", "Koklass Pheasant", "Lady Amherst's Pheasant", "Indian Pitta", "Pied Kingfisher", "Black-Capped Kingfisher", "Common Kingfisher", "Collared Kingfisher", "Stork-Billed Kingfisher", "White-Throated Kingfisher", "Ruddy Kingfisher", "Pied Kingfisher", "Himalayan Snow Leopard", "Clouded Leopard", "Bengal Fox", "Golden Jackal", "Gray Wolf", "Himalayan Wolf", "Red Fox", "Coyote", "Dholes", "Asiatic Black Bear", "Grizzly Bear", "Polar Bear", "Sloth Bear", "Red Panda", "Himalayan Musk Deer", "Jungle Cat", "Leopard Cat", "Snow Leopard", "Bengal Cat", "Tibetan Sand Fox", "Tibetan Wolf", "Wild Yak", "Wild Bison", "Wild Buffalo", "Wild Boar", "Himalayan Tahr", "Nilgai", "Goral", "Serow", "Muntjac", "Hog Deer", "Sambar", "Chital", "Barasingha", "Blackbuck", "Kashmir Stag", "Snow Leopard", "Bengal Tiger", "Indian Rhinoceros", "One-Horned Rhinoceros", "Gaur", "Elephant", "Asiatic Elephant", "Pygmy Elephant", "Sumatran Elephant", "African Elephant", "Asiatic Lion", "Snow Leopard", "Leopard", "Cheetah", "Jaguar", "Panther", "Cougar", "Lion", "Himalayan Black Bear", "Grizzly Bear", "Polar Bear", "Sloth Bear", "Red Panda", "Pangolin", "Civet", "Otter", "Badger", "Honey Badger", "Raccoon", "Skunk", "Fox", "Gray Fox", "Red Fox", "Kit Fox", "Arctic Fox", "Hedgehog", "Squirrel", "Golden-Bellied Squirrel", "Red Giant Flying Squirrel", "Himalayan Giant Squirrel", "Pangolin", "Flying Squirrel", "Tarsier", "Macaque", "Langur", "Hanuman Langur", "Bonnet Macaque", "Lion-Tailed Macaque", "Golden Langur", "Phayre's Langur", "Rhesus Macaque", "Baboon", "Mangabey", "Gibbon", "Hoolock Gibbon", "Agile Gibbon", "Siamang", "Orangutan", "Gorilla", "Chimpanzee", "Bactrian Camel", "Dromedary Camel", "Wild Ass", "Kiang", "Tibetan Wild Ass", "Wild Horse", "Przewalski's Horse", "Yak", "Banteng", "Water Buffalo", "Gaur", "Wild Bison", "Goral", "Serow", "Nilgai", "Muntjac", "Hog Deer", "Sambar", "Chital", "Barasingha", "Blackbuck", "Kashmir Stag", "Musk Deer", "Markhor", "Bharal", "Thar", "Naur", "Maral", "Kiang", "Argali", "Snow Leopard", "Leopard", "Lion", "Tiger", "Jaguar", "Cougar", "Puma", "Snow Leopard", "Clouded Leopard", "Marbled Cat", "Leopard Cat", "Jungle Cat", "Wild Boar", "Pig", "Aardvark", "Anteater", "Sloth", "Tapir", "Manatee", "Dugong", "Whale Shark", "Hammerhead Shark", "Bull Shark", "Tiger Shark", "Great White Shark", "Rays", "Manta Ray", "Stingray", "Humpback Whale", "Gray Whale", "Blue Whale", "Fin Whale", "Minke Whale", "Humpback Dolphin", "Bottlenose Dolphin", "Common Dolphin", "Spotted Dolphin", "Striped Dolphin", "Risso's Dolphin", "Spinner Dolphin", "Humpback Dolphin", "Pacific White-Sided Dolphin", "Neem", "Tulsi", "Lotus", "Bamboo", "Banyan", "Peepal", "Sandalwood", "Rose", "Jasmine", "Lemon Grass", "Aloe Vera", "Mango", "Banana", "Guava", "Jackfruit", "Coconut", "Papaya", "Cashew", "Pomegranate", "Grapes", "Starfruit", "Apple", "Litchi", "Mangosteen", "Longan", "Fig", "Custard Apple", "Bael", "Apricot", "Peach", "Plum", "Kiwi", "Persimmon", "Walnut", "Pistachio", "Almond", "Date Palm", "Blackberry", "Raspberry", "Strawberry", "Blueberry", "Black Currant", "Gooseberry", "Tomato", "Eggplant", "Potato", "Garlic", "Onion", "Carrot", "Pumpkin", "Squash", "Zucchini", "Gourd", "Radish", "Turnip", "Raddish", "Mustard", "Cauliflower", "Broccoli", "Cabbage", "Kale", "Spinach", "Swiss Chard", "Lettuce", "Endive", "Collard", "Escarole", "Arugula", "Radicchio", "Chicory", "Fennel", "Dill", "Coriander", "Parsley", "Cumin", "Mustard Seeds", "Sesame Seeds", "Anise", "Caraway", "Fennel Seeds", "Celery", "Asparagus", "Green Beans", "Lima Beans", "Peas", "Chickpeas", "Lentils", "Mung Beans", "Kidney Beans", "Black-eyed Peas", "Adzuki Beans", "Soybeans", "Peanuts", "Hazelnuts", "Chestnuts", "Walnuts", "Pine Nuts", "Macadamia Nuts", "Almonds", "Brazils", "Pecans", "Cashews", "Coconuts", "Sugar Cane", "Rice", "Wheat", "Barley", "Oats", "Maize", "Sorghum", "Millet", "Buckwheat", "Quinoa", "Amaranth", "Rye", "Triticale", "Cannabis", "Hemp", "Marijuana", "Tobacco", "Opium Poppy", "Cocoa", "Coffee", "Tea", "Pepper", "Ginger", "Turmeric", "Cinnamon", "Cloves", "Cardamom", "Fenugreek", "Mustard", "Coriander", "Cumin", "Fennel", "Anise", "Caraway", "Sage", "Thyme", "Oregano", "Rosemary", "Basil", "Parsley", "Dill", "Cilantro", "Mint", "Lavender", "Marjoram", "Bay Leaves", "Curry Leaves", "Betel", "Areca", "Kava", "Khat", "Mitragyna", "Salvia", "Ephedra", "Yohimbine", "Ginseng", "Licorice", "Bacopa", "Ashwagandha", "Turmeric", "Guggulu", "Shilajit", "Gymnema", "Brahmi", "Amalaki", "Haritaki", "Bibhitaki", "Guduchi", "Shatavari", "Vidari", "Guggulu", " Shankhapushpi", "Vacha", "Jatamansi", "Tagara", "Manjistha", "Nirgundi", "Gokshura", "Ashwagandha", "Ginger", "Garlic", "Turmeric", "Cumin", "Fennel", "Cloves", "Cardamom", "Cinnamon", "Mustard Seeds", "Black Pepper", "Long Pepper", "Betel Leaf", "Tamarind", "Lemon Grass", "Lemon Verbena", "Citronella", "Lemongrass", "Mint", "Peppermint", "Basil", "Rosemary", "Thyme", "Oregano", "Sage", "Bay Leaves", "Lavender", "Marjoram", "Cilantro", "Coriander", "Dill", "Parsley", "Chamomile", "Calendula", "Lemon Balm", "Echinacea", "Ginseng", "Gingko Biloba", "Gotu kola", "Saw Palmetto", "St. John's Wort", "Ginger Root", "Turmeric Root", "Garlic Bulbs", "Red Sandalwood", "White Sandalwood", "Yellow Sandalwood", "Eucalyptus", "Tea Tree", "Lemon Eucalyptus", "Peppermint Oil", "Lavender Oil", "Eucalyptus Oil", "Tea Tree Oil", "Alpaca", "Anaconda", "Andean Condor", "Andean Goose", "Andean Hillstar", "Andean Mountain Cat", "Andean Nightjar", "Andean Parakeet", "Andean Tinamou", "Andean White-tailed Deer", "Armadillo", "Atlantic Humpback Dolphin", "Atlantic Saddleback Dolphin", "Azara's Agouti", "Baboon Spiders", "Bald Uakari", "Ball Python", "Bare-tailed Woolly Opossum", "Bats", "Bearded Tegu", "Beaver", "Beluga Whale", "Bengal Tiger", "Bicolored Parakeet", "Black Caiman", "Black Howler Monkey", "Black Jaguar", "Black Rhinoceros", "Black Spiny-tailed Iguana", "Black Squirrel Monkey", "Black Vulture", "Blunt-snouted Bully", "Boa Constrictor", "Bolivian Squirrel", "Bondegezou", "Brazilian Aye-aye", "Brazilian Dolphin", "Brazilian Merganser", "Brazilian Squirrel", "Brown Capuchin Monkey", "Brown Howler Monkey", "Brown Pelican", "Brown-throated Three-toed Sloth", "Brown-tufted Capuchin", "Buffy-headed Marmoset", "Bulldog Bat", "Bullfrog", "Bushbaby", "Bushbaby Bat", "Butterfly Bat", "Butterfly Fish", "Byron’s Beaked Whale", "Caimans", "Calico Bass", "Cape Falcon", "Cape Horn Racer", "Capybara", "Caracal", "Central American Squirrel Monkey", "Cetaceans", "Chacoan Peccary", "Chilean Flamingo", "Chilean Fox", "Chilean Rose Tarantula", "Chimpanzee", "Chinchilla", "Cinereous Harrier", "Cinnamon Teal", "Clawed Otter", "Coati", "Collared Peccary", "Common Dolphin", "Common Opossum", "Common Squirrel Monkey", "Common Vampire Bat", "Cone-nosed Beetle", "Coral", "Cougar", "Crab-eating Fox", "Crowned Solitary Eagle", "Crowned Woodnymph", "Crowned-hunting Falcon", "Crying Pony", "Dark-winged Trumpeter", "Deer", "Desmarest's Tufted-tailed Rat", "Desmodus rotundus", "Dolphin", "Donkey", "Dormouse", "Double-striped Thick-knee", "Dusky Dolphin", "Dusky-legged Guan", "Eagle", "Eagle Ray", "Earless Seal", "Eared Dove", "Echidnas", "Emerald Tree Boa", "Emperor Tamarin", "Falkland Islands Dolphin", "False Vampire Bat", "Fat-tailed Dwarf Lemur", "Fer-de-Lance", "Fishers", "Flamingos", "Four-eyed Opossum", "Gaur", "Geoffroy's Tamarin", "Giant Anteater", "Giant Armadillo", "Giant Brazilian Otter", "Giant Otter", "Giant River Otter", "Giant South American Turtle", "Giant Swordfish", "Giant Tiger Shrimp", "Giant Wombat", "Giant Yellow Bat", "Gibbon", "Giraffe", "Golden-spectacled Bear", "Golden-tailed Sapphire", "Golden-whiskered Bat", "Goliath Bird-eating Spider", "Goliath Heron", "Goliath Tiger Fish", "Goral", "Great Horned Owl", "Green Anaconda", "Green Iguanas", "Green Sea Turtle", "Grey Dolphin", "Grey Fox", "Grey Seal", "Grey Whales", "Guinea Pig", "Guppy", "Guppy Fish", "Guppy Fry", "Harp Seal", "Harvest Mouse", "Hawaiian Monk Seal", "Hippopotamus", "Hook-billed Kite", "Howler Monkey", "Huemul", "Hunt's Antelope Squirrel", "Hunting Dog", "Hydrochoerus hydrochaeris", "Iberian Lynx", "Imperial Parrot", "Inca Dove", "Indian Elephant", "Indian Muntjac", "Inland Beaked Whale", "Jaguar", "Jaguarundi", "Joanna's Sapphire", "Joanna's Tamarin", "Keel-billed Toucan", "Killer Whale", "King Vulture", "Kingfisher", "Kinkajou", "Kookaburra", "Lama", "Lammergeier", "Largemouth Bass", "Leatherback Sea Turtle", "Lesser Yellow-headed Vulture", "Llama", "Long-tailed Brush-furred Rat", "Loon", "Macaw", "Maned Wolf", "Marine Otter", "Marsh Deer", "Masked Booby", "Mediterranean Monk Seal", "Mesopotamian Fallow Deer", "Mexican Long-tongued Bat", "Mexican Tamandua", "Mink", "Miscanthus Monkey", "Mitred Parakeet", "Mole", "Mole Rat", "Monk Seal", "Montevideo Marmoset", "Moon Jellyfish", "Moorish Idols", "Mountain Lion", "Mouse", "Musk Deer", "Musk Ox", "Narwhal", "Neotropical Otter", "Neotropical River Otter", "Neotropical Squirrel", "Neotropical Tree Rat", "Neotropical Trumpeter", "Neotropical White-crowned Manakin", "New World Monkey", "Nightingale", "North American Black Bear", "North American Bison", "North American River Otter", "North Atlantic Right Whale", "Northern Beaked Whale", "Northern Fur Seal", "Northern River Otter", "Norwegian Forest Cat", "Numbat", "Ocean Sunfish", "Octopus", "Okapi", "Old World Monkey", "Olingo", "Onager", "Opossum", "Orange-throated Sunangel", "Orca", "Ocelot", "Olinguito", "Otter", "Ocelot", "Pampas Cat", "Pangolin", "Parrot", "Patagonian Hare", "Peccary", "Pelican", "Peruvian Pelican", "Puma", "Piranha", "Piton", "Plains Vizcacha", "Platypus", "Porcupine", "Puma", "Puraque", "Purple Gallinule", "Pygmy Marmoset", "Pygmy Owl", "Rainbow Boa", "Rainbow Trout", "Rat", "Red Brocket", "Red Howler Monkey", "Red Squirrel Monkey", "Angel's trumpet", "Orchid Tree", "Andean Potato Tree", "Chilca", "Lupine", "South American Cactus", "Purple Passionflower", "Yellow Trumpet", "Amazon Lily", "Amazon Sword Plant", "Aperea", "Armadillo Gourd", "Baccharis", "Bignonia", "Brazilian Water Lily", "Bromeliads", "Cacti", "Calycophoran", "Canavalia", "Canary Creeper", "Cassava", "Castor Bean", "Celosia", "Chile Peppers", "Chuchuhuasi", "Cocona", "Copaiba", "Corn", "Costus", "Cotton", "Crotalaria", "Croton", "Cryptanthus", "Cucumber Tree", "Cyperus", "Daisy", "Datura", "Desmodium", "Devil's Claw", "Dillenia", "Dodder", "Echinodorus", "Echium", "Elephant Ear", "Eucalyptus", "Eupatorium", "Ferns", "Flame Vine", "Flax", "Fuchsia", "Gentian", "Giant Coneflower", "Ginger", "Golden Trumpet", "Gourds", "Grass", "Groundcherry", "Guava", "Hollyhock", "Honokioli", "Ilex", "Impatiens", "Indigo", "Ironweed", "Jacaranda", "Jasmine", "Jojoba", "Kapok Tree", "Kava", "Lantana", "Lavender", "Lemon Verbena", "Lettuce", "Lupin", "Magnolia", "Malvaceae", "Maple", "Mauritius Hemp", "Mimosa", "Monkey Brush", "Monstera", "Mulberry", "Narcissus", "Nasturtium", "Nettles", "Nightshade", "Okra", "Old Man's Beard", "Oleander", "Oregano", "Passionflower", "Pepper", "Peruvian Lily", "Philodendron", "Physalis", "Pineapple", "Pitahaya", "Pitcher Plant", "Poinsettia", "Pokeweed", "Pomegranate", "Poppy", "Pothos", "Purslane", "Queen of the Night", "Radish", "Rhododendron", "Rue", "Russian Olive", "Sage", "Saintpaulia", "Salicornia", "Sassafras", "Scarlet Sage", "Screwpine", "Senna", "Sesbania", "Sesuvium", "Sisal", "Sorrel", "Spathiphyllum", "Stachytarpheta", "Star Fruit", "Strelitzia", "Sugar Cane", "Sunflower", "Sweet Potato", "Tomato", "Tulip", "Ugni", "Umbrella Plant", "Urtica", "Vervain", "Vinca", "Wattle", "Wax Plant", "Willow", "Wisteria", "Yarrow", "Zinnia","Amazon Rainforest, Brazil", "Great Barrier Reef, Australia", "Yellowstone National Park, USA", "Serengeti National Park, Tanzania", "Victoria Falls, Zambia/Zimbabwe", "Mount Everest, Nepal", "Angel Falls, Venezuela", "Mount Kilimanjaro, Tanzania", "Bora Bora, French Polynesia", "Galapagos Islands, Ecuador", "Maldives", "The Grand Canyon, USA", "Iguazu Falls, Argentina/Brazil", "Niagara Falls, USA/Canada", "Cape of Good Hope, South Africa", "Santorini, Greece", "Sistine Chapel, Vatican City", "Halong Bay, Vietnam", "Taj Mahal, India", "Mount Tai, China", "Mount Fuji, Japan", "Bali, Indonesia", "The Great Wall of China", "Red Square, Russia", "Acropolis of Athens, Greece", "The Colosseum, Italy", "The Leaning Tower of Pisa, Italy", "Stonehenge, England", "The Pyramids of Giza, Egypt", "Machu Picchu, Peru", "Uyuni Salt Flats, Bolivia", "Jokulsarlon Glacier Lagoon, Iceland", "Plitvice Lakes National Park, Croatia", "Yosemite National Park, USA", "Denali National Park, Alaska", "Banff National Park, Canada", "Kruger National Park, South Africa", "Serengeti National Park, Tanzania", "Tasmanian Wilderness, Australia", "The Great Smoky Mountains, USA", "Glacier National Park, USA", "Death Valley National Park, USA", "Zion National Park, USA", "Everglades National Park, USA", "Rocky Mountain National Park, USA", "Yellowstone National Park, USA", "Rainbow Mountains, China", "Wadi Rum, Jordan", "Mount Roraima, Venezuela/Brazil/Guyana", "Mount Fanjing, China", "The Matterhorn, Switzerland", "Mount Kilimanjaro, Tanzania", "Aletsch Glacier, Switzerland", "The Lost City of Petra, Jordan", "Mount Everest Base Camp, Nepal", "Mount Toubkal, Morocco", "The Andes, South America", "The Swiss Alps, Switzerland", "Mount Kilimanjaro, Tanzania", "The Himalayas, Nepal/Tibet", "The Rocky Mountains, USA/Canada", "The Canadian Rockies, Canada", "The Alps, Europe", "The Dolomites, Italy", "The Pyrenees, France/Spain", "The Urals, Russia", "Mount Rainier, USA", "Mount St. Helens, USA", "Mount Shasta, USA", "Mount Hood, USA", "Mount Adams, USA", "Mount Baker, USA", "Mount Jefferson, USA", "Mount Olympus, Greece", "Mount Elbrus, Russia", "Mount Blanc, France/Italy", "Mount Shkhara, Georgia", "Mount Ararat, Turkey", "Mount Damavand, Iran", "Mount Aconcagua, Argentina", "Mount Kilimanjaro, Tanzania", "Mount Denali, Alaska", "Mount Vinson, Antarctica", "Mount Kosciuszko, Australia", "Mount Wilhelm, Papua New Guinea", "Mount Chimborazo, Ecuador", "Mount Meru, Tanzania", "Mount Yasur, Vanuatu", "Mount Stromboli, Italy", "Mount Vesuvius, Italy", "Mount Etna, Italy", "Mount Kilimanjaro, Tanzania", "Mount Mayon, Philippines", "Mount Taal, Philippines","Mount Popocatepetl, Mexico", "Mount Süphan, Turkey", "Mount Damavand, Iran", "Mount Rinjani, Indonesia", "Mount Bromo, Indonesia", "Mount Fuji, Japan", "Mount Aso, Japan", "Mount Ontake, Japan", "Mount Merbabu, Indonesia", "Mount Semeru, Indonesia", "Mount Shasta, USA", "Mount Edziza, Canada", "Mount Cayley, Canada", "Mount Baker, USA", "Mount Aspiring, New Zealand", "Mount Cook, New Zealand", "Mount Tasman, New Zealand", "Mount Puncak Jaya, Indonesia", "Mount Kinabalu, Malaysia", "Mount Fansipan, Vietnam", "Mount Langbiang, Vietnam", "Mount Hua Shan, China", "Mount Songshan, China", "Mount Emei, China", "Mount Wutai, China", "Mount Everest Base Camp Trek, Nepal", "Mount Toubkal National Park, Morocco", "Mount Tari, Papua New Guinea", "Mount Wilhelm, Papua New Guinea", "Mount Kinabalu National Park, Malaysia", "Mount Kilimanjaro National Park, Tanzania", "Mount Everest National Park, Nepal", "Mount Kinangop, Kenya", "Mount Tomboly, Papua New Guinea", "Mount Elgon National Park, Uganda/Kenya", "Mount Shikaribetsu, Japan", "Mount Myohyang, North Korea", "Mount Sobaeksan, South Korea", "Mount Osorezan, Japan", "Mount Zao, Japan", "Mount Hakusan, Japan", "Mount Tamalpais, USA", "Mount Monadnock, USA", "Mount Greylock, USA", "Mount Washington, USA", "Mount Katahdin, USA", "Mount St. Helens National Volcanic Monument, USA", "Mount Rainier National Park, USA", "Mount Baker-Snoqualmie National Forest, USA", "Mount Hood National Forest, USA", "Mount Adams Wilderness, USA", "Mount Shasta Wilderness, USA", "Mount Hood Wilderness, USA", "Mount Rainier Wilderness, USA", "Mount Everest Extreme, Nepal", "Mount Kilimanjaro Climb, Tanzania", "Mount Everest Base Camp and Gokyo Lakes Trek, Nepal", "Mount Kilimanjaro Marangu Route, Tanzania", "Mount Kilimanjaro Rongai Route, Tanzania", "Mount Kilimanjaro Machame Route, Tanzania", "Mount Kilimanjaro Lemosho Route, Tanzania", "Yellowstone National Park", "The Great Barrier Reef", "Victoria Falls", "Mount Kilimanjaro", "The Amazon Rainforest", "The Galapagos Islands", "Mount Fuji", "Glacier National Park", "The Serengeti", "Mount Denali", "The Yosemite Valley", "Mount Rainier", "Banff National Park", "The Rocky Mountains", "The Andes", "The Swiss Alps", "Mount Saint Helens", "The Yosemite High Country", "The Great Smoky Mountains", "Crater Lake", "Mount Shasta", "The Hudson River Valley", "The Blue Ridge Mountains", "The Appalachian Trail", "Mount Hood", "The Sierra Nevada", "The Adirondacks", "Mount Washington", "The Grand Tetons", "The Catskills", "Mount St. Helens National Volcanic Monument", "The Mohonk Preserve", "Mount Adams", "Mount Baker", "Mount Bachelor", "Mount Blane", "Mount Bona", "Mount Byers", "Mount Cleveland", "Mount Columbia", "Mount Shavano", "Mount Sneffels", "Mount Wilson", "Mount Rainier National Park", "Mount Rushmore National Memorial", "Mount Hood National Forest", "Mount Baker-Snoqualmie National Forest", "Mount Adams Wilderness", "Mount Rainier Wilderness", "Mount St. Helens Wilderness", "Mount Hood Wilderness", "Mount Shasta Wilderness", "Mount Sneffels Wilderness", "Yellowstone National Park Wilderness", "The Grand Canyon National Park Wilderness", "Glacier National Park Wilderness", "Yosemite National Park Wilderness", "The Great Smoky Mountains National Park Wilderness", "The Appalachian National Scenic Trail", "The Pacific Crest Trail", "The Continental Divide Trail", "The John Muir Trail", "The Colorado Trail", "The Tahoe Rim Trail", "The Wonderland Trail", "The Knobstone Trail", "The Bruce Trail", "The Pacific Rim National Park Reserve", "The Cape Breton Highlands National Park", "The Fundy National Park", "The Jasper National Park", "The Banff National Park", "The Kootenay National Park", "The Waterton Lakes National Park", "The Yoho National Park", "The Glacier National Park", "The Mount Revelstoke National Park", "The Nahanni National Park Reserve", "The Gwaii Haanas National Park Reserve", "The Kejimkujik National Park", "The Forillon National Park", "The Jasper National Park of Canada", "The Bruce Peninsula", "The Fundy Trail Parkway", "The Thousand Islands National Park", "The Algonquin Provincial Park", "The Banff National Park of Canada", "The Point Pelee National Park", "The La Mauricie National Park", "The Cape Breton Highlands National Park of Canada", "The Pacific Rim National Park Reserve of Canada", "The Rocky Mountain National Park", "The Dinosaur Provincial Park", "The Canyons of the Ancients National Monument", "The Chiricahua National Monument", "The Bandelier National Monument", "The Aztec Ruins National Monument", "The Chaco Culture National Historical Park", "The Glacier Bay National Park and Preserve", "The Wrangell-St. Elias National Park and Preserve", "The Lake Clark National Park and Preserve", "The Gates of the Arctic National Park and Preserve", "The Kenai Fjords National Park", "The Klondike Gold Rush National Historical Park", "The Misty Fjords National Monument", "The Wrangell-St. Elias Wilderness", "The Lake Clark Wilderness", "The Yosemite Wilderness", "The Redwood National and State Parks", "The Channel Islands National Park", "The Sequoia National Park", "The Kings Canyon National Park", "The Bryce Canyon National Park", "The Zion National Park", "The Arches National Park", "The Canyonlands National Park", "The Joshua Tree National Park", "The Petrified Forest National Park", "The Saguaro National Park", "The Badlands National Park", "The Black Canyon of the Gunnison National Park", "The Grand Teton National Park", "The Great Sand Dunes National Park and Preserve", "The Isle Royale National Park", "The Dry Tortugas National Park", "The Everglades National Park", "The Acadia National Park", "The Bryce Canyon National Park of Utah", "The Big Bend National Park", "The Guadalupe Mountains National Park", "The Congaree National Park", "The Great Basin National Park", "The Redwood National Park of California", "The Kings Canyon National Park of California", "The Yosemite National Park of California", "The Bryce Canyon National Park of Utah", "The Zion National Park of Utah", "The Arches National Park of Utah", "The Canyonlands National Park of Utah", "The Joshua Tree National Park of California", "The Olympic National Park", "The North Cascades National Park", "The Wind Cave National Park", "The Badlands National Park of South Dakota", "The Black Canyon of the Gunnison National Park of Colorado", "The Rocky Mountain National Park of Colorado", "The Petrified Forest National Park of Arizona", "The Saguaro National Park of Arizona", "The Great Sand Dunes National Park and Preserve of Colorado", "The Isle Royale National Park of Michigan", "The Dry Tortugas National Park of Florida", "The Everglades National Park of Florida", "The Acadia National Park of Maine", "The Congaree National Park of South Carolina", "The Great Basin National Park of Nevada", "The Guadalupe Mountains National Park of Texas", "The Big Bend National Park of Texas", "The Grand Teton National Park of Wyoming", "The Yosemite National Park of California", "The Redwood National Park of California", "The Kings Canyon National Park of California", "The Sequoia National Park of California", "The Channel Islands National Park of California", "The Bryce Canyon National Park of Utah", "The Zion National Park of Utah", "The Arches National Park of Utah", "The Canyonlands National Park of Utah", "The Joshua Tree National Park of California", "Rainforest," "Tropical Trees," "Jungle Ecosystem," "Flora and Fauna," "Wildlife," "Amazon River," "Wildflowers," "Endangered Species," "Mammals," "Birds," "Reptiles," "Amphibians," "Insects," "Marine Life," "Beaches," "Ocean Currents," "Coral Reefs," "Marine Plants," "Seagulls," "Dolphins," "Whales," "Sharks," "Krill," "Plankton," "Sea Turtles," "Starfish," "Jellyfish," "Algae," "Seashells," "Sand Dunes," "Tides," "Deserts," "Cacti," "Desert Wildlife," "Desert Flowers," "Mesas," "Ravines," "Arroyos," "Hot Springs," "Geysers," "Volcanoes," "Mountains," "Rock Formations," "Canyons," "Gorges," "Rivers," "Lakes," "Waterfalls," "Streams," "Swamps," "Marshes," "Wetlands," "Bogs," "Fens," "Mangroves," "Salt Marshes," "Forests," "Trees," "Bushes," "Shrubs," "Evergreens," "Deciduous Trees," "Conifers," "Ferns," "Mosses," "Lianas," "Vines," "Grasses," "Herbs," "Wild Berries," "Edible Wild Plants," "Poisonous Plants," "Mushrooms," "Fungi," "Lichens," "Bacteria," "Algae," "Protists," "Plants," "Flowers," "Tropical Fruits," "Berries," "Coconuts," "Mangoes," "Papayas," "Bananas," "Avocados," "Pineapples," "Kiwis," "Lemons," "Limes," "Oranges," "Grapes," "Apples," "Pears," "Peaches," "Plums," "Cherries," "Raspberries," "Strawberries," "Blueberries," "Blackberries," "Cranberries," "Eucalyptus," "Redwoods," "Maple Trees," "Oak Trees," "Willow Trees," "Birch Trees," "Elm Trees," "Beech Trees," "Cedar Trees," "Spruce Trees," "Pine Trees," "Fir Trees," "Hemlock Trees," "Larch Trees," "Yew Trees," "Juniper Trees," "Cypress Trees," "Dogwood Trees," "Sassafras Trees," "Magnolia Trees," "Chestnut Trees," "Walnut Trees," "Pecan Trees," "Hickory Trees," "Oak Trees," "Maple Trees."]


MODE_VIDEO_STREAM = False

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
        self.video_link = ""
        
        self.start_time = 0
        self.session_token = None
        self.offline = False
        self.is_download_video_running = False

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
            # self.offline = True
            if not self.offline:
                root.update_idletasks()
                root.update()
                        
                if self.keyword_reset:
                    i = len(organisms)
                    self.keyword = organisms[random.randint(0, i-1)]
                    print("USING KEYWORD: {}".format(self.keyword))
                    self.keyword_reset = False            

                if not self.get_video_ready and not self.get_video_running:
                    print("ONLINE: RUNNING GET VIDEO")
                    self.get_video() #threaded
                
                if not self.get_chatgpt_text_ready and not self.get_chatgpt_text_running:
                    print("ONLINE: RUNNING CHATGPT") 
                    self.get_chatgpt_text("a 5 paragraph text on {}, narrated by david attenborough. use some funny nature jokes.".format((self.keyword).strip())) #threaded

                if not self.is_inference_ready and self.get_chatgpt_text_ready and not self.is_inference_running:
                    print("ONLINE: RUNNING INFERENCE")
                    self.run_inference() #threaded

                if self.get_video_ready and self.get_chatgpt_text_ready and self.is_inference_ready and not self.is_play_audio_running:       
                    print("ONLINE: STARTING SHOW")          
                    self.play_audio() # threaded
                    time.sleep(2)
                    self.play_main_video()      
                time.sleep(1)

            else:
                root.update_idletasks()
                root.update()
                #offline mode, get stuff from files
                
                if not self.get_video_ready and not self.get_video_running:
                    print("OFFLINE: RUNNING GET VIDEO")
                    self.get_video() #threaded
                
                if not self.is_inference_ready and not self.is_inference_running:
                    print("OFFLINE: RUNNING INFERENCE")
                    self.run_inference() #threaded

                if self.get_video_ready and self.is_inference_ready and not self.is_play_audio_running: 
                    print("OFFLINE: STARTING SHOW")               
                    self.play_audio() # threaded
                    time.sleep(2)
                    self.play_main_video()      
                time.sleep(1)

            
    def play(self):
        #copy ready video into main video
        if not MODE_VIDEO_STREAM:
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
        # while True:
        #     if self.get_chatgpt_text_ready and self.get_video_ready and self.is_inference_ready:
        #         break
        #     time.sleep(0.5)

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
        self.t_play = threading.Thread(target=self.play, args=())
        self.t_play.start()

    def run_inference(self):
        self.is_inference_running = True
        self.t_inf = threading.Thread(target=self.run_inference_t, args=())
        self.t_inf.start()

    def get_chatgpt_text(self, text):
        self.get_chatgpt_text_running = True
        self.t_gpt = threading.Thread(target=self.get_chatgpt_text_t, args=(text,))
        self.t_gpt.start()

    def get_video(self):
        if self.offline:         
            i = 0
            while True:
                filename = os.path.join("texts/", str(i) + ".txt")
                if not os.path.exists(filename):
                    break
                i += 1
            r = random.randint(0, i)
            f = "texts/{}.txt".format(r)
            print("RUNNING VIDEO SEARCH OFFLINE: GETTING FILE {}".format(f))
            with open(f, 'r') as f:
                self.keyword = f.readline().strip()
                self.chatgpt_message = f.read().strip()

        self.get_video_running = True
        self.t_get_video = threading.Thread(target=self.get_video_link_and_download_t, args=())
        self.t_get_video.start()        
   
    def get_video_link_and_download_t(self):

        if MODE_VIDEO_STREAM:        
            while True:
                try:
                    videosSearch = None
                    while not videosSearch:
                        videosSearch = VideosSearch(self.keyword, limit = 20) #not enough?
                        time.sleep(0.5)       

                    # check durations and find best
                    
                    results = videosSearch.result()
                    filtered = []
                    for x in results['result']:
                        if not "none" in x['duration']:
                            d = x['duration'].split(':')[0]
                            if int(d) <= 10 and int(d) >= 3:    
                                filtered.append(x)

                    r = random.randint(0,len(filtered)-1)
                except:
                    print("VIDEO SEARCH FAILED, trying again with different keyword!")
                    i = len(organisms)
                    self.keyword = "nature {}".format(self.keyword)
                    time.sleep(0.5)
                else:
                    break
            print("video search done")
            #get video title
            self.video_title = filtered[r]['title']
            self.video_link = filtered[r]['link'] 
        else:
            if os.path.exists("prep_video.mp4"):
                os.remove('prep_video.mp4')
            
            while not os.path.exists("prep_video.mp4"):
                while True:
                    try:
                        videosSearch = None
                        while not videosSearch:
                            videosSearch = VideosSearch(self.keyword, limit = 20) #not enough?
                            time.sleep(0.5)       

                        # check durations and find best
                        
                        results = videosSearch.result()
                        filtered = []
                        for x in results['result']:
                            if not "none" in x['duration']:
                                d = x['duration'].split(':')[0]
                                if int(d) <= 10 and int(d) >= 3:    
                                    filtered.append(x)

                        r = random.randint(0,len(filtered)-1)
                    except:
                        print("Video search failed, trying again with different keyword!")
                        i = len(organisms)
                        self.keyword = organisms[random.randint(0, i-1)]
                        time.sleep(0.5)
                    else:
                        break

                print("video search done")

                #get video title
                self.video_title = filtered[r]['title']

                link =  filtered[r]['link'] 
                #return results['result'][0]['link']

                # os.system("yt-dlp {} -f 137 -o main_video.%(ext)s".format(link))
                # os.system("yt-dlp {} -f 136 -o prep_video.%(ext)s".format(link))
                self.is_download_video_running = True
                os.system("yt-dlp {} -f bv*[ext=mp4] --max-filesize 400M -o prep_video.%(ext)s".format(link))
                self.is_download_video_running = False
                time.sleep(1)

        self.get_video_ready = True
        self.get_video_running = False

    def get_chatgpt_text_t(self, text):

        # #check timer for rate limit
        # while (time.time() - self.start_time) < 300:
        #     print("Waiting for rate limit timer")
        #     print((time.time() - self.start_time))
        #     time.sleep(5)
        # self.start_timer()

        self.session_token = 'eyJhbGciOiJkaXIiLCJlbmMiOiJBMjU2R0NNIn0..yrDB2Ni-wsZs_XO6.djJ2xe7jUrxTOYXpu5gogTBmmcw2b_lC-1Msxu4Chh4XdcLPQHtY66AndVhthqjVNaL-kzIu6DugUeIuO_g0wK5acYGWdbm24yPl18H-szA7fWicl6-ZVEiNfl3JcuLGYR7boTdQ8tPp6z_jqacPDns8w0lDNEAhWk09C5OUMKmEyohKHgP3rgEBpa_tKjutJzLIcQUe-zhgCg7iBoA38yUGlVZuSlwiP7FMGdvk6dFMNZczZ4iOOjDdsRxmaFxta23MKBBL-A7M5Af3Ocesf4TgIb2PH8nQfYPfbsdcacAoWKxcIuIh2vHzxigb8Ma-lb1Vh2_lTWFKCfJ-0vQKsu5cqhaIUap6f1E-gt_nYAsQ6XF847nq-lrinKqvHQEZPQLy_seozpYba0FpPOYj44wfmGxQSqGJd7S75tjNVLPnJolqP9i-NXrWeuTcUgiO_Q7pNlZS8hV0nx_c1n15hCoUAN3dqsjZ--_aRnT4e1UkWt1W1BoYG46dgiBcuLhcccuYEXLqJqIex6DVGn2gt0buMrNhyzzsnsGcbU5AgXCqoPxr8ZhRZgiI1-3UfonPGEta-rOWO16e3N4MtfVKexhlcvncE2DwhqzDTnQCRJr7AZ5qGB6rJz6Akov0ywyQ0RyL1rmGbUrh6H2WH1T0suCZpahf6RIOcQY35rb1QdoH4CTjqud-seedByKcr6znQMjL-fe6RUtiQzaBwOYMYxuOYy5G0Z8R_hA6gVqVsy3Z5x9Yw1b0uIGSpE3XSkTUCjdzmng8SE9ydzCkcczaLFbV57M5b4vQbPwZR8EhMd6qDX-v_KoDlbWGIeFfp3KfGKYVbokMHRbm0bJgeggFJPwhjOb-lvKjojvPJDEBgd4cMlWml2I1KLLYhwN8C3DPJZqeTxfZwAgVpjxQQ5ScYqPiG7kTcM-te8vmUDsTMWX5vXjDlJKyhyIOMu5OfO3mqo3-No40zG8aNMQulUTX_XCVE8vSbIti5Qs87VjhS70ZAaSXo8QK5TjKT9VerID7Bl4u1I1CMXMiYLdk4fZhydSG4lKvHamOYSAKg-RL0DpbAG6Y3c-ftPFL7DKKqpjAAUm4HS1_eW52CoE0q3_ON_Yk_gTrXwLSS8oOfaZ_48rAPoHEcLODezD-xA4SdffBFtIynM9YGR5QBmiKWyPTX_X_K_9MLkYS4vzP-UHp6HIJctoLoztD_71wVhP6eOpybKUtPSD-8dFLdef3wXjLCGJKO3mx0G07icrit56StANR0VzFLRlaBkwxekk-74ec1X8vVzCTARjh-t5P02HD7kR9Z_Od9LKZ5dUnG1pgVXKSV87cdInLuU1pVcVlbEENec-Mc6ES2UQu3SWG8qBkziFHUl0nyUb4_8R4FBRy-sepba-zoN_UveLrLbLUn1LDCuKMRsYYwOUUPYWffcHil4fuan1j9hGZZsUJ9Dt0LmiByp9Ng-zW1UFOzt8uy-cUnl9FkdStC2-AvGKZ-PquRnuDpZcqsJ0pkEWwsBi9byoKYBSAcy259cHPA6mgxGljuv8K0IhQtJwdf-mOcrHgKJmikSM4yN9uB5eOcyLQB7a98GckUPwQRSWojyI6-lVTaKQIDH7em-PerKqj7X-WOeNjH5FTlk7ChuI7TzifXFOtyefbkf_wGMfu9CN_A1IaWbuzuYE71m4Tn_V0QyYJz8Pkhj9HAFrv_zTRcYNnTP8gUW745txwFUPTW_0wqQkuMmtJ3VoM2xmgJ81F3QavEy4VGC22nGz2RwuN0YQwDmRLo3rpWkEk5L6vxxe-Ek4B_BW9B79_MBbe2lXG0Ac5x9qAadZb7z-AKpfF_4VFi5g4ge9poYGJSlKTbkOaNgdDgTz2dhA786tv5ZmUiC6Yom4J0aKrVRQgxtAZvQTbQ4rERku02ATTW3fbu-srzbRzDxVWfUFUsrpePFTDpK0XJIdBrEeTeMAQiD4jwiG1TqP-jDKYlmUByNjuxXeKaHqzm6IKIXmDKr4ld3_UaE_cbZJfszsW7u8ZyhZ-CJwnCPUDcLVnntj2BR3r0JGOweNAuIISBJks5r6jM4pWDrBhPZ4G9T4Q5eK4vImUvmYVhgoaCuEF7Ux6LuE3U8f5Ym2EQL8FfCyCNPhlY1BjxSRicdpudnqU2cxdmxqW7Tb41IZonVYy6E9oh93X6SwEH37kASoXvxOb3rJ7b7T6ZsjQTiScA7HJeBZfs174TP_Bq7tnFCm35t0lHmL61Wwtc-ob1m9JpUruiI0lZ1jaksCWEitbVgbpS6o6eo7w-xjr1hzosQ9_pUpxmjpzpfbixILHnH_y6dxXcruJCYWIEXkAYeUpfgVcnYX9oOTApQLyyVO6KhY3i3ovmzsyhbuBrBUsC8LJDs0nTmcgINi5M9Ss53Z4kEIxIHlJvErmYE-NjOrD6Sc1yHbKWsSqtmXpESdEXKipGY5thAvvNOf-Zta7TGmXdUwf_RTnrcDw3Zojrhzlz8UyuA2JnjDocoDs8y89G_prTWtZ6Yj2pvDhxyoTEZXrpihEydwYIILXSjXSMpOAFXM9dNa_4kq0bmTpXEm2VkNu_IqXljnau7k-CVvIHQoDo2sViXewmh6vPkuXlKZYVgEmswvf3Qd-hcQNlPxzcuIhSefXz2VZV_fvu0Y6f2BuyfnVHUOs219MBr-ruVkPaxfbbQEtlV0yE_o.vTYR3P2e8MVXamfAOvAjew' 
        
        
        api = ChatGPT(self.session_token)
        
        first_run = True
        while True:
            try:
                print("Sending GPT message")
                t = api.send_message(text)
                ####
            except:
                print("RUNNING OFFLINE")
                # re initiallize youtube search and download
                if first_run:
                    self.is_inference_ready = False
                    self.is_inference_running = False
                    self.get_chatgpt_text_ready = False
                    self.get_chatgpt_text_running = False
                    self.get_video_ready = False
                    self.get_video_running = False
                    self.is_play_audio_running = False
                    self.keyword_reset = True
                    first_run = False

                    #wait for video downloader to reset
                    while self.is_download_video_running:
                        time.sleep(1)

                # self.get_chatgpt_text_ready = False
                # self.get_chatgpt_text_running = False

                self.offline = True
                api.__del__()
                api = ChatGPT(self.session_token)      
            else:
                self.offline = False
                break
            time.sleep(3600)
            # self.offline = True
            # self.get_chatgpt_text_ready = False
            # self.get_chatgpt_text_running = False
            # print("RUNNING OFFLINE")
            # time.sleep(60)

        #api.reset_conversation()
        self.chatgpt_message = t['message']
        # t = "Good day to all, it's David Attenborough here. Today, I would like to talk to you about one of the most fascinating creatures in the animal kingdom - the Wombat. Wombats are short-legged, sturdy and heavy creatures with a reputation for being slow-moving and docile."

        #self.chatgpt_message = t

        #write message to file
        subdir = "texts"
        if not os.path.exists(subdir):
            os.makedirs(subdir)
        i = 0
        while True:
            filename = os.path.join(subdir, str(i) + ".txt")
            if not os.path.exists(filename):
                break
            i += 1
        filename = "texts/{}.txt".format(i)
        with open(filename, 'w') as f:
            f.write("{}\n".format(self.keyword))
            f.write(self.chatgpt_message)

        api.__del__()
        self.get_chatgpt_text_ready = True
        self.get_chatgpt_text_running = False

    def play_main_video(self):
        #audio playing ahead in first offline vid
        if MODE_VIDEO_STREAM:
            # creating pafy object of the video 
            video = pafy.new(self.video_link)             
            # getting best stream 
            best = video.getbest()             
            # creating vlc media player object 
            vlcmedia = self.vlc_obj.media_new(best.url)
            self.vlcplayer.set_media(vlcmedia)
        else:
            vlcmedia = self.vlc_obj.media_new("main_video.mp4")
            self.vlcplayer.set_media(vlcmedia)
        self.vlcplayer.set_hwnd(video_frame.winfo_id())#tkinter label or frame

        self.vlcplayer.audio_set_mute(True)
        self.vlcplayer.play()

        self.title_label = tk.Label(root, text="{}:: {}".format(self.keyword, self.video_title), font=("TkDefaultFont", 15), bg='black', fg='white')
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
       
       # make 2 lines into one for better inference
       
            #Split text 
            text = self.chatgpt_message
            text = text.strip('\n')
            text = text.strip('"')
            text = text.strip('”')
            text = text.strip('“')            
            text_list = re.split(r'(?<=[\.\!\?])\s*', text)

            #remove blank and short cuts
            text_list_cleaned = []
            x = 0
            joined_text = ""
            for i in text_list:
                if i and len(i) > 10:
                    if x == 0:
                        joined_text = i
                    else:
                        joined_text = "{} {}".format(joined_text, i)
                        text_list_cleaned.append(joined_text)
                        joined_text = ""
                        x = 0
                    x += 1
            if joined_text:
                text_list_cleaned.append(joined_text)

            # text_list = [i for i in text_list if i]
            print(text_list_cleaned)
            wav_list = []

            for i, t in enumerate(text_list_cleaned):
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

