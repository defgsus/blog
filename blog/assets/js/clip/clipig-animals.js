document.addEventListener( "DOMContentLoaded", function() {

    const IMAGES = {
        "accordion": {
            "cat": "cat-of-accordion",
            "elephant": "elephant-of-accordion",
            "hedgehog": "hedgehog-of-accordion",
            "kangaroo": "kangaroo-of-accordion",
            "peacock": "peacock-of-accordion",
            "penguin": "penguin-of-accordion",
            "pig": "pig-of-accordion",
            "snail": "snail-of-accordion",
            "tapir": "tapir-of-accordion"
        },
        "apple": {
            "cat": "cat-of-apple",
            "elephant": "elephant-of-apple",
            "hedgehog": "hedgehog-of-apple",
            "kangaroo": "kangaroo-of-apple",
            "peacock": "peacock-of-apple",
            "penguin": "penguin-of-apple",
            "pig": "pig-of-apple",
            "snail": "snail-of-apple",
            "tapir": "tapir-of-apple"
        },
        "avocado": {
            "cat": "cat-of-avocado",
            "elephant": "elephant-of-avocado",
            "hedgehog": "hedgehog-of-avocado",
            "kangaroo": "kangaroo-of-avocado",
            "peacock": "peacock-of-avocado",
            "penguin": "penguin-of-avocado",
            "pig": "pig-of-avocado",
            "snail": "snail-of-avocado",
            "tapir": "tapir-of-avocado"
        },
        "baguette": {
            "cat": "cat-of-baguette",
            "elephant": "elephant-of-baguette",
            "hedgehog": "hedgehog-of-baguette",
            "kangaroo": "kangaroo-of-baguette",
            "peacock": "peacock-of-baguette",
            "penguin": "penguin-of-baguette",
            "pig": "pig-of-baguette",
            "snail": "snail-of-baguette",
            "tapir": "tapir-of-baguette"
        },
        "banana": {
            "cat": "cat-of-banana",
            "elephant": "elephant-of-banana",
            "hedgehog": "hedgehog-of-banana",
            "kangaroo": "kangaroo-of-banana",
            "peacock": "peacock-of-banana",
            "penguin": "penguin-of-banana",
            "pig": "pig-of-banana",
            "snail": "snail-of-banana",
            "tapir": "tapir-of-banana"
        },
        "basil": {
            "cat": "cat-of-basil",
            "elephant": "elephant-of-basil",
            "hedgehog": "hedgehog-of-basil",
            "kangaroo": "kangaroo-of-basil",
            "peacock": "peacock-of-basil",
            "penguin": "penguin-of-basil",
            "pig": "pig-of-basil",
            "snail": "snail-of-basil",
            "tapir": "tapir-of-basil"
        },
        "beetroot": {
            "cat": "cat-of-beetroot",
            "elephant": "elephant-of-beetroot",
            "hedgehog": "hedgehog-of-beetroot",
            "kangaroo": "kangaroo-of-beetroot",
            "peacock": "peacock-of-beetroot",
            "penguin": "penguin-of-beetroot",
            "pig": "pig-of-beetroot",
            "snail": "snail-of-beetroot",
            "tapir": "tapir-of-beetroot"
        },
        "broom": {
            "cat": "cat-of-broom",
            "elephant": "elephant-of-broom",
            "hedgehog": "hedgehog-of-broom",
            "kangaroo": "kangaroo-of-broom",
            "peacock": "peacock-of-broom",
            "penguin": "penguin-of-broom",
            "pig": "pig-of-broom",
            "snail": "snail-of-broom",
            "tapir": "tapir-of-broom"
        },
        "burger": {
            "cat": "cat-of-burger",
            "elephant": "elephant-of-burger",
            "hedgehog": "hedgehog-of-burger",
            "kangaroo": "kangaroo-of-burger",
            "peacock": "peacock-of-burger",
            "penguin": "penguin-of-burger",
            "pig": "pig-of-burger",
            "snail": "snail-of-burger",
            "tapir": "tapir-of-burger"
        },
        "burrito": {
            "cat": "cat-of-burrito",
            "elephant": "elephant-of-burrito",
            "hedgehog": "hedgehog-of-burrito",
            "kangaroo": "kangaroo-of-burrito",
            "peacock": "peacock-of-burrito",
            "penguin": "penguin-of-burrito",
            "pig": "pig-of-burrito",
            "snail": "snail-of-burrito",
            "tapir": "tapir-of-burrito"
        },
        "cabbage": {
            "cat": "cat-of-cabbage",
            "elephant": "elephant-of-cabbage",
            "hedgehog": "hedgehog-of-cabbage",
            "kangaroo": "kangaroo-of-cabbage",
            "peacock": "peacock-of-cabbage",
            "penguin": "penguin-of-cabbage",
            "pig": "pig-of-cabbage",
            "snail": "snail-of-cabbage",
            "tapir": "tapir-of-cabbage"
        },
        "cake": {
            "cat": "cat-of-cake",
            "elephant": "elephant-of-cake",
            "hedgehog": "hedgehog-of-cake",
            "kangaroo": "kangaroo-of-cake",
            "peacock": "peacock-of-cake",
            "penguin": "penguin-of-cake",
            "pig": "pig-of-cake",
            "snail": "snail-of-cake",
            "tapir": "tapir-of-cake"
        },
        "calamari": {
            "cat": "cat-of-calamari",
            "elephant": "elephant-of-calamari",
            "hedgehog": "hedgehog-of-calamari",
            "kangaroo": "kangaroo-of-calamari",
            "peacock": "peacock-of-calamari",
            "penguin": "penguin-of-calamari",
            "pig": "pig-of-calamari",
            "snail": "snail-of-calamari",
            "tapir": "tapir-of-calamari"
        },
        "carpet": {
            "cat": "cat-of-carpet",
            "elephant": "elephant-of-carpet",
            "hedgehog": "hedgehog-of-carpet",
            "kangaroo": "kangaroo-of-carpet",
            "peacock": "peacock-of-carpet",
            "penguin": "penguin-of-carpet",
            "pig": "pig-of-carpet",
            "snail": "snail-of-carpet",
            "tapir": "tapir-of-carpet"
        },
        "carrot": {
            "cat": "cat-of-carrot",
            "elephant": "elephant-of-carrot",
            "hedgehog": "hedgehog-of-carrot",
            "kangaroo": "kangaroo-of-carrot",
            "peacock": "peacock-of-carrot",
            "penguin": "penguin-of-carrot",
            "pig": "pig-of-carrot",
            "snail": "snail-of-carrot",
            "tapir": "tapir-of-carrot"
        },
        "chili": {
            "cat": "cat-of-chili",
            "elephant": "elephant-of-chili",
            "hedgehog": "hedgehog-of-chili",
            "kangaroo": "kangaroo-of-chili",
            "peacock": "peacock-of-chili",
            "penguin": "penguin-of-chili",
            "pig": "pig-of-chili",
            "snail": "snail-of-chili",
            "tapir": "tapir-of-chili"
        },
        "coconut": {
            "cat": "cat-of-coconut",
            "elephant": "elephant-of-coconut",
            "hedgehog": "hedgehog-of-coconut",
            "kangaroo": "kangaroo-of-coconut",
            "peacock": "peacock-of-coconut",
            "penguin": "penguin-of-coconut",
            "pig": "pig-of-coconut",
            "snail": "snail-of-coconut",
            "tapir": "tapir-of-coconut"
        },
        "coldness": {
            "cat": "cat-of-coldness",
            "elephant": "elephant-of-coldness",
            "hedgehog": "hedgehog-of-coldness",
            "kangaroo": "kangaroo-of-coldness",
            "peacock": "peacock-of-coldness",
            "penguin": "penguin-of-coldness",
            "pig": "pig-of-coldness",
            "snail": "snail-of-coldness",
            "tapir": "tapir-of-coldness"
        },
        "comfort": {
            "cat": "cat-of-comfort",
            "elephant": "elephant-of-comfort",
            "hedgehog": "hedgehog-of-comfort",
            "kangaroo": "kangaroo-of-comfort",
            "peacock": "peacock-of-comfort",
            "penguin": "penguin-of-comfort",
            "pig": "pig-of-comfort",
            "snail": "snail-of-comfort",
            "tapir": "tapir-of-comfort"
        },
        "coral reef": {
            "cat": "cat-of-coral-reef",
            "elephant": "elephant-of-coral-reef",
            "hedgehog": "hedgehog-of-coral-reef",
            "kangaroo": "kangaroo-of-coral-reef",
            "peacock": "peacock-of-coral-reef",
            "penguin": "penguin-of-coral-reef",
            "pig": "pig-of-coral-reef",
            "snail": "snail-of-coral-reef",
            "tapir": "tapir-of-coral-reef"
        },
        "corkscrew": {
            "cat": "cat-of-corkscrew",
            "elephant": "elephant-of-corkscrew",
            "hedgehog": "hedgehog-of-corkscrew",
            "kangaroo": "kangaroo-of-corkscrew",
            "peacock": "peacock-of-corkscrew",
            "penguin": "penguin-of-corkscrew",
            "pig": "pig-of-corkscrew",
            "snail": "snail-of-corkscrew",
            "tapir": "tapir-of-corkscrew"
        },
        "croissant": {
            "cat": "cat-of-croissant",
            "elephant": "elephant-of-croissant",
            "hedgehog": "hedgehog-of-croissant",
            "kangaroo": "kangaroo-of-croissant",
            "peacock": "peacock-of-croissant",
            "penguin": "penguin-of-croissant",
            "pig": "pig-of-croissant",
            "snail": "snail-of-croissant",
            "tapir": "tapir-of-croissant"
        },
        "cuckoo clock": {
            "cat": "cat-of-cuckoo-clock",
            "elephant": "elephant-of-cuckoo-clock",
            "hedgehog": "hedgehog-of-cuckoo-clock",
            "kangaroo": "kangaroo-of-cuckoo-clock",
            "peacock": "peacock-of-cuckoo-clock",
            "penguin": "penguin-of-cuckoo-clock",
            "pig": "pig-of-cuckoo-clock",
            "snail": "snail-of-cuckoo-clock",
            "tapir": "tapir-of-cuckoo-clock"
        },
        "cucumber": {
            "cat": "cat-of-cucumber",
            "elephant": "elephant-of-cucumber",
            "hedgehog": "hedgehog-of-cucumber",
            "kangaroo": "kangaroo-of-cucumber",
            "peacock": "peacock-of-cucumber",
            "penguin": "penguin-of-cucumber",
            "pig": "pig-of-cucumber",
            "snail": "snail-of-cucumber",
            "tapir": "tapir-of-cucumber"
        },
        "cursive letters": {
            "cat": "cat-of-cursive-letters",
            "elephant": "elephant-of-cursive-letters",
            "hedgehog": "hedgehog-of-cursive-letters",
            "kangaroo": "kangaroo-of-cursive-letters",
            "peacock": "peacock-of-cursive-letters",
            "penguin": "penguin-of-cursive-letters",
            "pig": "pig-of-cursive-letters",
            "snail": "snail-of-cursive-letters",
            "tapir": "tapir-of-cursive-letters"
        },
        "cymbals": {
            "cat": "cat-of-cymbals",
            "elephant": "elephant-of-cymbals",
            "hedgehog": "hedgehog-of-cymbals",
            "kangaroo": "kangaroo-of-cymbals",
            "peacock": "peacock-of-cymbals",
            "penguin": "penguin-of-cymbals",
            "pig": "pig-of-cymbals",
            "snail": "snail-of-cymbals",
            "tapir": "tapir-of-cymbals"
        },
        "eggplant": {
            "cat": "cat-of-eggplant",
            "elephant": "elephant-of-eggplant",
            "hedgehog": "hedgehog-of-eggplant",
            "kangaroo": "kangaroo-of-eggplant",
            "peacock": "peacock-of-eggplant",
            "penguin": "penguin-of-eggplant",
            "pig": "pig-of-eggplant",
            "snail": "snail-of-eggplant",
            "tapir": "tapir-of-eggplant"
        },
        "eraser": {
            "cat": "cat-of-eraser",
            "elephant": "elephant-of-eraser",
            "hedgehog": "hedgehog-of-eraser",
            "kangaroo": "kangaroo-of-eraser",
            "peacock": "peacock-of-eraser",
            "penguin": "penguin-of-eraser",
            "pig": "pig-of-eraser",
            "snail": "snail-of-eraser",
            "tapir": "tapir-of-eraser"
        },
        "faucet": {
            "cat": "cat-of-faucet",
            "elephant": "elephant-of-faucet",
            "hedgehog": "hedgehog-of-faucet",
            "kangaroo": "kangaroo-of-faucet",
            "peacock": "peacock-of-faucet",
            "penguin": "penguin-of-faucet",
            "pig": "pig-of-faucet",
            "snail": "snail-of-faucet",
            "tapir": "tapir-of-faucet"
        },
        "fossil": {
            "cat": "cat-of-fossil",
            "elephant": "elephant-of-fossil",
            "hedgehog": "hedgehog-of-fossil",
            "kangaroo": "kangaroo-of-fossil",
            "peacock": "peacock-of-fossil",
            "penguin": "penguin-of-fossil",
            "pig": "pig-of-fossil",
            "snail": "snail-of-fossil",
            "tapir": "tapir-of-fossil"
        },
        "fried chicken": {
            "cat": "cat-of-fried-chicken",
            "elephant": "elephant-of-fried-chicken",
            "hedgehog": "hedgehog-of-fried-chicken",
            "kangaroo": "kangaroo-of-fried-chicken",
            "peacock": "peacock-of-fried-chicken",
            "penguin": "penguin-of-fried-chicken",
            "pig": "pig-of-fried-chicken",
            "snail": "snail-of-fried-chicken",
            "tapir": "tapir-of-fried-chicken"
        },
        "fries": {
            "cat": "cat-of-fries",
            "elephant": "elephant-of-fries",
            "hedgehog": "hedgehog-of-fries",
            "kangaroo": "kangaroo-of-fries",
            "peacock": "peacock-of-fries",
            "penguin": "penguin-of-fries",
            "pig": "pig-of-fries",
            "snail": "snail-of-fries",
            "tapir": "tapir-of-fries"
        },
        "garlic": {
            "cat": "cat-of-garlic",
            "elephant": "elephant-of-garlic",
            "hedgehog": "hedgehog-of-garlic",
            "kangaroo": "kangaroo-of-garlic",
            "peacock": "peacock-of-garlic",
            "penguin": "penguin-of-garlic",
            "pig": "pig-of-garlic",
            "snail": "snail-of-garlic",
            "tapir": "tapir-of-garlic"
        },
        "geiger counter": {
            "cat": "cat-of-geiger-counter",
            "elephant": "elephant-of-geiger-counter",
            "hedgehog": "hedgehog-of-geiger-counter",
            "kangaroo": "kangaroo-of-geiger-counter",
            "peacock": "peacock-of-geiger-counter",
            "penguin": "penguin-of-geiger-counter",
            "pig": "pig-of-geiger-counter",
            "snail": "snail-of-geiger-counter",
            "tapir": "tapir-of-geiger-counter"
        },
        "glacier": {
            "cat": "cat-of-glacier",
            "elephant": "elephant-of-glacier",
            "hedgehog": "hedgehog-of-glacier",
            "kangaroo": "kangaroo-of-glacier",
            "peacock": "peacock-of-glacier",
            "penguin": "penguin-of-glacier",
            "pig": "pig-of-glacier",
            "snail": "snail-of-glacier",
            "tapir": "tapir-of-glacier"
        },
        "gourd": {
            "cat": "cat-of-gourd",
            "elephant": "elephant-of-gourd",
            "hedgehog": "hedgehog-of-gourd",
            "kangaroo": "kangaroo-of-gourd",
            "peacock": "peacock-of-gourd",
            "penguin": "penguin-of-gourd",
            "pig": "pig-of-gourd",
            "snail": "snail-of-gourd",
            "tapir": "tapir-of-gourd"
        },
        "grater": {
            "cat": "cat-of-grater",
            "elephant": "elephant-of-grater",
            "hedgehog": "hedgehog-of-grater",
            "kangaroo": "kangaroo-of-grater",
            "peacock": "peacock-of-grater",
            "penguin": "penguin-of-grater",
            "pig": "pig-of-grater",
            "snail": "snail-of-grater",
            "tapir": "tapir-of-grater"
        },
        "harmonica": {
            "cat": "cat-of-harmonica",
            "elephant": "elephant-of-harmonica",
            "hedgehog": "hedgehog-of-harmonica",
            "kangaroo": "kangaroo-of-harmonica",
            "peacock": "peacock-of-harmonica",
            "penguin": "penguin-of-harmonica",
            "pig": "pig-of-harmonica",
            "snail": "snail-of-harmonica",
            "tapir": "tapir-of-harmonica"
        },
        "harp": {
            "cat": "cat-of-harp",
            "elephant": "elephant-of-harp",
            "hedgehog": "hedgehog-of-harp",
            "kangaroo": "kangaroo-of-harp",
            "peacock": "peacock-of-harp",
            "penguin": "penguin-of-harp",
            "pig": "pig-of-harp",
            "snail": "snail-of-harp",
            "tapir": "tapir-of-harp"
        },
        "hospital": {
            "cat": "cat-of-hospital",
            "elephant": "elephant-of-hospital",
            "hedgehog": "hedgehog-of-hospital",
            "kangaroo": "kangaroo-of-hospital",
            "peacock": "peacock-of-hospital",
            "penguin": "penguin-of-hospital",
            "pig": "pig-of-hospital",
            "snail": "snail-of-hospital",
            "tapir": "tapir-of-hospital"
        },
        "kettle": {
            "cat": "cat-of-kettle",
            "elephant": "elephant-of-kettle",
            "hedgehog": "hedgehog-of-kettle",
            "kangaroo": "kangaroo-of-kettle",
            "peacock": "peacock-of-kettle",
            "penguin": "penguin-of-kettle",
            "pig": "pig-of-kettle",
            "snail": "snail-of-kettle",
            "tapir": "tapir-of-kettle"
        },
        "lettuce": {
            "cat": "cat-of-lettuce",
            "elephant": "elephant-of-lettuce",
            "hedgehog": "hedgehog-of-lettuce",
            "kangaroo": "kangaroo-of-lettuce",
            "peacock": "peacock-of-lettuce",
            "penguin": "penguin-of-lettuce",
            "pig": "pig-of-lettuce",
            "snail": "snail-of-lettuce",
            "tapir": "tapir-of-lettuce"
        },
        "loaf of bread": {
            "cat": "cat-of-loaf-of-bread",
            "elephant": "elephant-of-loaf-of-bread",
            "hedgehog": "hedgehog-of-loaf-of-bread",
            "kangaroo": "kangaroo-of-loaf-of-bread",
            "peacock": "peacock-of-loaf-of-bread",
            "penguin": "penguin-of-loaf-of-bread",
            "pig": "pig-of-loaf-of-bread",
            "snail": "snail-of-loaf-of-bread",
            "tapir": "tapir-of-loaf-of-bread"
        },
        "lotus root": {
            "cat": "cat-of-lotus-root",
            "elephant": "elephant-of-lotus-root",
            "hedgehog": "hedgehog-of-lotus-root",
            "kangaroo": "kangaroo-of-lotus-root",
            "peacock": "peacock-of-lotus-root",
            "penguin": "penguin-of-lotus-root",
            "pig": "pig-of-lotus-root",
            "snail": "snail-of-lotus-root",
            "tapir": "tapir-of-lotus-root"
        },
        "lychee": {
            "cat": "cat-of-lychee",
            "elephant": "elephant-of-lychee",
            "hedgehog": "hedgehog-of-lychee",
            "kangaroo": "kangaroo-of-lychee",
            "peacock": "peacock-of-lychee",
            "penguin": "penguin-of-lychee",
            "pig": "pig-of-lychee",
            "snail": "snail-of-lychee",
            "tapir": "tapir-of-lychee"
        },
        "mango": {
            "cat": "cat-of-mango",
            "elephant": "elephant-of-mango",
            "hedgehog": "hedgehog-of-mango",
            "kangaroo": "kangaroo-of-mango",
            "peacock": "peacock-of-mango",
            "penguin": "penguin-of-mango",
            "pig": "pig-of-mango",
            "snail": "snail-of-mango",
            "tapir": "tapir-of-mango"
        },
        "mangosteen": {
            "cat": "cat-of-mangosteen",
            "elephant": "elephant-of-mangosteen",
            "hedgehog": "hedgehog-of-mangosteen",
            "kangaroo": "kangaroo-of-mangosteen",
            "peacock": "peacock-of-mangosteen",
            "penguin": "penguin-of-mangosteen",
            "pig": "pig-of-mangosteen",
            "snail": "snail-of-mangosteen",
            "tapir": "tapir-of-mangosteen"
        },
        "maple leaf": {
            "cat": "cat-of-maple-leaf",
            "elephant": "elephant-of-maple-leaf",
            "hedgehog": "hedgehog-of-maple-leaf",
            "kangaroo": "kangaroo-of-maple-leaf",
            "peacock": "peacock-of-maple-leaf",
            "penguin": "penguin-of-maple-leaf",
            "pig": "pig-of-maple-leaf",
            "snail": "snail-of-maple-leaf",
            "tapir": "tapir-of-maple-leaf"
        },
        "meatloaf": {
            "cat": "cat-of-meatloaf",
            "elephant": "elephant-of-meatloaf",
            "hedgehog": "hedgehog-of-meatloaf",
            "kangaroo": "kangaroo-of-meatloaf",
            "peacock": "peacock-of-meatloaf",
            "penguin": "penguin-of-meatloaf",
            "pig": "pig-of-meatloaf",
            "snail": "snail-of-meatloaf",
            "tapir": "tapir-of-meatloaf"
        },
        "motorcycle": {
            "cat": "cat-of-motorcycle",
            "elephant": "elephant-of-motorcycle",
            "hedgehog": "hedgehog-of-motorcycle",
            "kangaroo": "kangaroo-of-motorcycle",
            "peacock": "peacock-of-motorcycle",
            "penguin": "penguin-of-motorcycle",
            "pig": "pig-of-motorcycle",
            "snail": "snail-of-motorcycle",
            "tapir": "tapir-of-motorcycle"
        },
        "mushroom": {
            "cat": "cat-of-mushroom",
            "elephant": "elephant-of-mushroom",
            "hedgehog": "hedgehog-of-mushroom",
            "kangaroo": "kangaroo-of-mushroom",
            "peacock": "peacock-of-mushroom",
            "penguin": "penguin-of-mushroom",
            "pig": "pig-of-mushroom",
            "snail": "snail-of-mushroom",
            "tapir": "tapir-of-mushroom"
        },
        "orange": {
            "cat": "cat-of-orange",
            "elephant": "elephant-of-orange",
            "hedgehog": "hedgehog-of-orange",
            "kangaroo": "kangaroo-of-orange",
            "peacock": "peacock-of-orange",
            "penguin": "penguin-of-orange",
            "pig": "pig-of-orange",
            "snail": "snail-of-orange",
            "tapir": "tapir-of-orange"
        },
        "parsnip": {
            "cat": "cat-of-parsnip",
            "elephant": "elephant-of-parsnip",
            "hedgehog": "hedgehog-of-parsnip",
            "kangaroo": "kangaroo-of-parsnip",
            "peacock": "peacock-of-parsnip",
            "penguin": "penguin-of-parsnip",
            "pig": "pig-of-parsnip",
            "snail": "snail-of-parsnip",
            "tapir": "tapir-of-parsnip"
        },
        "peace": {
            "cat": "cat-of-peace",
            "elephant": "elephant-of-peace",
            "hedgehog": "hedgehog-of-peace",
            "kangaroo": "kangaroo-of-peace",
            "peacock": "peacock-of-peace",
            "penguin": "penguin-of-peace",
            "pig": "pig-of-peace",
            "snail": "snail-of-peace",
            "tapir": "tapir-of-peace"
        },
        "peach": {
            "cat": "cat-of-peach",
            "elephant": "elephant-of-peach",
            "hedgehog": "hedgehog-of-peach",
            "kangaroo": "kangaroo-of-peach",
            "peacock": "peacock-of-peach",
            "penguin": "penguin-of-peach",
            "pig": "pig-of-peach",
            "snail": "snail-of-peach",
            "tapir": "tapir-of-peach"
        },
        "piano": {
            "cat": "cat-of-piano",
            "elephant": "elephant-of-piano",
            "hedgehog": "hedgehog-of-piano",
            "kangaroo": "kangaroo-of-piano",
            "peacock": "peacock-of-piano",
            "penguin": "penguin-of-piano",
            "pig": "pig-of-piano",
            "snail": "snail-of-piano",
            "tapir": "tapir-of-piano"
        },
        "pickle": {
            "cat": "cat-of-pickle",
            "elephant": "elephant-of-pickle",
            "hedgehog": "hedgehog-of-pickle",
            "kangaroo": "kangaroo-of-pickle",
            "peacock": "peacock-of-pickle",
            "penguin": "penguin-of-pickle",
            "pig": "pig-of-pickle",
            "snail": "snail-of-pickle",
            "tapir": "tapir-of-pickle"
        },
        "pie": {
            "cat": "cat-of-pie",
            "elephant": "elephant-of-pie",
            "hedgehog": "hedgehog-of-pie",
            "kangaroo": "kangaroo-of-pie",
            "peacock": "peacock-of-pie",
            "penguin": "penguin-of-pie",
            "pig": "pig-of-pie",
            "snail": "snail-of-pie",
            "tapir": "tapir-of-pie"
        },
        "pineapple": {
            "cat": "cat-of-pineapple",
            "elephant": "elephant-of-pineapple",
            "hedgehog": "hedgehog-of-pineapple",
            "kangaroo": "kangaroo-of-pineapple",
            "peacock": "peacock-of-pineapple",
            "penguin": "penguin-of-pineapple",
            "pig": "pig-of-pineapple",
            "snail": "snail-of-pineapple",
            "tapir": "tapir-of-pineapple"
        },
        "pizza": {
            "cat": "cat-of-pizza",
            "elephant": "elephant-of-pizza",
            "hedgehog": "hedgehog-of-pizza",
            "kangaroo": "kangaroo-of-pizza",
            "peacock": "peacock-of-pizza",
            "penguin": "penguin-of-pizza",
            "pig": "pig-of-pizza",
            "snail": "snail-of-pizza",
            "tapir": "tapir-of-pizza"
        },
        "plate": {
            "cat": "cat-of-plate",
            "elephant": "elephant-of-plate",
            "hedgehog": "hedgehog-of-plate",
            "kangaroo": "kangaroo-of-plate",
            "peacock": "peacock-of-plate",
            "penguin": "penguin-of-plate",
            "pig": "pig-of-plate",
            "snail": "snail-of-plate",
            "tapir": "tapir-of-plate"
        },
        "polygons": {
            "cat": "cat-of-polygons",
            "elephant": "elephant-of-polygons",
            "hedgehog": "hedgehog-of-polygons",
            "kangaroo": "kangaroo-of-polygons",
            "peacock": "peacock-of-polygons",
            "penguin": "penguin-of-polygons",
            "pig": "pig-of-polygons",
            "snail": "snail-of-polygons",
            "tapir": "tapir-of-polygons"
        },
        "potato chip": {
            "cat": "cat-of-potato-chip",
            "elephant": "elephant-of-potato-chip",
            "hedgehog": "hedgehog-of-potato-chip",
            "kangaroo": "kangaroo-of-potato-chip",
            "peacock": "peacock-of-potato-chip",
            "penguin": "penguin-of-potato-chip",
            "pig": "pig-of-potato-chip",
            "snail": "snail-of-potato-chip",
            "tapir": "tapir-of-potato-chip"
        },
        "pretzel": {
            "cat": "cat-of-pretzel",
            "elephant": "elephant-of-pretzel",
            "hedgehog": "hedgehog-of-pretzel",
            "kangaroo": "kangaroo-of-pretzel",
            "peacock": "peacock-of-pretzel",
            "penguin": "penguin-of-pretzel",
            "pig": "pig-of-pretzel",
            "snail": "snail-of-pretzel",
            "tapir": "tapir-of-pretzel"
        },
        "raspberry": {
            "cat": "cat-of-raspberry",
            "elephant": "elephant-of-raspberry",
            "hedgehog": "hedgehog-of-raspberry",
            "kangaroo": "kangaroo-of-raspberry",
            "peacock": "peacock-of-raspberry",
            "penguin": "penguin-of-raspberry",
            "pig": "pig-of-raspberry",
            "snail": "snail-of-raspberry",
            "tapir": "tapir-of-raspberry"
        },
        "rosemary": {
            "cat": "cat-of-rosemary",
            "elephant": "elephant-of-rosemary",
            "hedgehog": "hedgehog-of-rosemary",
            "kangaroo": "kangaroo-of-rosemary",
            "peacock": "peacock-of-rosemary",
            "penguin": "penguin-of-rosemary",
            "pig": "pig-of-rosemary",
            "snail": "snail-of-rosemary",
            "tapir": "tapir-of-rosemary"
        },
        "rubik\u2019s cube": {
            "cat": "cat-of-rubik-s-cube",
            "elephant": "elephant-of-rubik-s-cube",
            "hedgehog": "hedgehog-of-rubik-s-cube",
            "kangaroo": "kangaroo-of-rubik-s-cube",
            "peacock": "peacock-of-rubik-s-cube",
            "penguin": "penguin-of-rubik-s-cube",
            "pig": "pig-of-rubik-s-cube",
            "snail": "snail-of-rubik-s-cube",
            "tapir": "tapir-of-rubik-s-cube"
        },
        "russian doll": {
            "cat": "cat-of-russian-doll",
            "elephant": "elephant-of-russian-doll",
            "hedgehog": "hedgehog-of-russian-doll",
            "kangaroo": "kangaroo-of-russian-doll",
            "peacock": "peacock-of-russian-doll",
            "penguin": "penguin-of-russian-doll",
            "pig": "pig-of-russian-doll",
            "snail": "snail-of-russian-doll",
            "tapir": "tapir-of-russian-doll"
        },
        "salami": {
            "cat": "cat-of-salami",
            "elephant": "elephant-of-salami",
            "hedgehog": "hedgehog-of-salami",
            "kangaroo": "kangaroo-of-salami",
            "peacock": "peacock-of-salami",
            "penguin": "penguin-of-salami",
            "pig": "pig-of-salami",
            "snail": "snail-of-salami",
            "tapir": "tapir-of-salami"
        },
        "sardines": {
            "cat": "cat-of-sardines",
            "elephant": "elephant-of-sardines",
            "hedgehog": "hedgehog-of-sardines",
            "kangaroo": "kangaroo-of-sardines",
            "peacock": "peacock-of-sardines",
            "penguin": "penguin-of-sardines",
            "pig": "pig-of-sardines",
            "snail": "snail-of-sardines",
            "tapir": "tapir-of-sardines"
        },
        "soda can": {
            "cat": "cat-of-soda-can",
            "elephant": "elephant-of-soda-can",
            "hedgehog": "hedgehog-of-soda-can",
            "kangaroo": "kangaroo-of-soda-can",
            "peacock": "peacock-of-soda-can",
            "penguin": "penguin-of-soda-can",
            "pig": "pig-of-soda-can",
            "snail": "snail-of-soda-can",
            "tapir": "tapir-of-soda-can"
        },
        "spiral": {
            "cat": "cat-of-spiral",
            "elephant": "elephant-of-spiral",
            "hedgehog": "hedgehog-of-spiral",
            "kangaroo": "kangaroo-of-spiral",
            "peacock": "peacock-of-spiral",
            "penguin": "penguin-of-spiral",
            "pig": "pig-of-spiral",
            "snail": "snail-of-spiral",
            "tapir": "tapir-of-spiral"
        },
        "submarine": {
            "cat": "cat-of-submarine",
            "elephant": "elephant-of-submarine",
            "hedgehog": "hedgehog-of-submarine",
            "kangaroo": "kangaroo-of-submarine",
            "peacock": "peacock-of-submarine",
            "penguin": "penguin-of-submarine",
            "pig": "pig-of-submarine",
            "snail": "snail-of-submarine",
            "tapir": "tapir-of-submarine"
        },
        "sushi": {
            "cat": "cat-of-sushi",
            "elephant": "elephant-of-sushi",
            "hedgehog": "hedgehog-of-sushi",
            "kangaroo": "kangaroo-of-sushi",
            "peacock": "peacock-of-sushi",
            "penguin": "penguin-of-sushi",
            "pig": "pig-of-sushi",
            "snail": "snail-of-sushi",
            "tapir": "tapir-of-sushi"
        },
        "taco": {
            "cat": "cat-of-taco",
            "elephant": "elephant-of-taco",
            "hedgehog": "hedgehog-of-taco",
            "kangaroo": "kangaroo-of-taco",
            "peacock": "peacock-of-taco",
            "penguin": "penguin-of-taco",
            "pig": "pig-of-taco",
            "snail": "snail-of-taco",
            "tapir": "tapir-of-taco"
        },
        "tamale": {
            "cat": "cat-of-tamale",
            "elephant": "elephant-of-tamale",
            "hedgehog": "hedgehog-of-tamale",
            "kangaroo": "kangaroo-of-tamale",
            "peacock": "peacock-of-tamale",
            "penguin": "penguin-of-tamale",
            "pig": "pig-of-tamale",
            "snail": "snail-of-tamale",
            "tapir": "tapir-of-tamale"
        },
        "tank": {
            "cat": "cat-of-tank",
            "elephant": "elephant-of-tank",
            "hedgehog": "hedgehog-of-tank",
            "kangaroo": "kangaroo-of-tank",
            "peacock": "peacock-of-tank",
            "penguin": "penguin-of-tank",
            "pig": "pig-of-tank",
            "snail": "snail-of-tank",
            "tapir": "tapir-of-tank"
        },
        "tarragon": {
            "cat": "cat-of-tarragon",
            "elephant": "elephant-of-tarragon",
            "hedgehog": "hedgehog-of-tarragon",
            "kangaroo": "kangaroo-of-tarragon",
            "peacock": "peacock-of-tarragon",
            "penguin": "penguin-of-tarragon",
            "pig": "pig-of-tarragon",
            "snail": "snail-of-tarragon",
            "tapir": "tapir-of-tarragon"
        },
        "tempura": {
            "cat": "cat-of-tempura",
            "elephant": "elephant-of-tempura",
            "hedgehog": "hedgehog-of-tempura",
            "kangaroo": "kangaroo-of-tempura",
            "peacock": "peacock-of-tempura",
            "penguin": "penguin-of-tempura",
            "pig": "pig-of-tempura",
            "snail": "snail-of-tempura",
            "tapir": "tapir-of-tempura"
        },
        "tissue": {
            "cat": "cat-of-tissue",
            "elephant": "elephant-of-tissue",
            "hedgehog": "hedgehog-of-tissue",
            "kangaroo": "kangaroo-of-tissue",
            "peacock": "peacock-of-tissue",
            "penguin": "penguin-of-tissue",
            "pig": "pig-of-tissue",
            "snail": "snail-of-tissue",
            "tapir": "tapir-of-tissue"
        },
        "tissue box": {
            "cat": "cat-of-tissue-box",
            "elephant": "elephant-of-tissue-box",
            "hedgehog": "hedgehog-of-tissue-box",
            "kangaroo": "kangaroo-of-tissue-box",
            "peacock": "peacock-of-tissue-box",
            "penguin": "penguin-of-tissue-box",
            "pig": "pig-of-tissue-box",
            "snail": "snail-of-tissue-box",
            "tapir": "tapir-of-tissue-box"
        },
        "toaster": {
            "cat": "cat-of-toaster",
            "elephant": "elephant-of-toaster",
            "hedgehog": "hedgehog-of-toaster",
            "kangaroo": "kangaroo-of-toaster",
            "peacock": "peacock-of-toaster",
            "penguin": "penguin-of-toaster",
            "pig": "pig-of-toaster",
            "snail": "snail-of-toaster",
            "tapir": "tapir-of-toaster"
        },
        "tomato": {
            "cat": "cat-of-tomato",
            "elephant": "elephant-of-tomato",
            "hedgehog": "hedgehog-of-tomato",
            "kangaroo": "kangaroo-of-tomato",
            "peacock": "peacock-of-tomato",
            "penguin": "penguin-of-tomato",
            "pig": "pig-of-tomato",
            "snail": "snail-of-tomato",
            "tapir": "tapir-of-tomato"
        },
        "tuba": {
            "cat": "cat-of-tuba",
            "elephant": "elephant-of-tuba",
            "hedgehog": "hedgehog-of-tuba",
            "kangaroo": "kangaroo-of-tuba",
            "peacock": "peacock-of-tuba",
            "penguin": "penguin-of-tuba",
            "pig": "pig-of-tuba",
            "snail": "snail-of-tuba",
            "tapir": "tapir-of-tuba"
        },
        "turnip": {
            "cat": "cat-of-turnip",
            "elephant": "elephant-of-turnip",
            "hedgehog": "hedgehog-of-turnip",
            "kangaroo": "kangaroo-of-turnip",
            "peacock": "peacock-of-turnip",
            "penguin": "penguin-of-turnip",
            "pig": "pig-of-turnip",
            "snail": "snail-of-turnip",
            "tapir": "tapir-of-turnip"
        },
        "violin": {
            "cat": "cat-of-violin",
            "elephant": "elephant-of-violin",
            "hedgehog": "hedgehog-of-violin",
            "kangaroo": "kangaroo-of-violin",
            "peacock": "peacock-of-violin",
            "penguin": "penguin-of-violin",
            "pig": "pig-of-violin",
            "snail": "snail-of-violin",
            "tapir": "tapir-of-violin"
        },
        "waffle": {
            "cat": "cat-of-waffle",
            "elephant": "elephant-of-waffle",
            "hedgehog": "hedgehog-of-waffle",
            "kangaroo": "kangaroo-of-waffle",
            "peacock": "peacock-of-waffle",
            "penguin": "penguin-of-waffle",
            "pig": "pig-of-waffle",
            "snail": "snail-of-waffle",
            "tapir": "tapir-of-waffle"
        },
        "warmth": {
            "cat": "cat-of-warmth",
            "elephant": "elephant-of-warmth",
            "hedgehog": "hedgehog-of-warmth",
            "kangaroo": "kangaroo-of-warmth",
            "peacock": "peacock-of-warmth",
            "penguin": "penguin-of-warmth",
            "pig": "pig-of-warmth",
            "snail": "snail-of-warmth",
            "tapir": "tapir-of-warmth"
        },
        "watermelon": {
            "cat": "cat-of-watermelon",
            "elephant": "elephant-of-watermelon",
            "hedgehog": "hedgehog-of-watermelon",
            "kangaroo": "kangaroo-of-watermelon",
            "peacock": "peacock-of-watermelon",
            "penguin": "penguin-of-watermelon",
            "pig": "pig-of-watermelon",
            "snail": "snail-of-watermelon",
            "tapir": "tapir-of-watermelon"
        },
        "xylophone": {
            "cat": "cat-of-xylophone",
            "elephant": "elephant-of-xylophone",
            "hedgehog": "hedgehog-of-xylophone",
            "kangaroo": "kangaroo-of-xylophone",
            "peacock": "peacock-of-xylophone",
            "penguin": "penguin-of-xylophone",
            "pig": "pig-of-xylophone",
            "snail": "snail-of-xylophone",
            "tapir": "tapir-of-xylophone"
        }
    };

    function on_thing_changed() {
        console.log("HERE");
        const thing = document.querySelector(".clipig-input .thing").value;
        const images = IMAGES[thing];
        for (const animal of Object.keys(images)) {
            const filename = `../../../assets/images/clipig/dall-e-samples/${images[animal]}.png`;
            const img_elem = document.querySelector(`.clipig-images .image-${animal}`);
            img_elem.src = filename;
        }
    }

    function render_interface(root_elem) {
        html = `<select class="thing">`
        for (const thing of Object.keys(IMAGES)) {
            html += `<option value="${thing}">${thing}</option>`;
        }
        html += `</select><hr>`;

        html += `<div class="clipig-images">`;
        for (const animal of Object.keys(IMAGES["salami"])) {
            html += `<div class="image-container">`;
            html += `<img class="image-${animal}">`;
            html += `<div class="image-text">${animal}</div></div>`;
        }
        html += `</div>`;

        root_elem.innerHTML = html;
        root_elem.querySelector(".thing").onchange = on_thing_changed;
    }

    render_interface(document.querySelector(".clipig-input"));
    on_thing_changed();

});
