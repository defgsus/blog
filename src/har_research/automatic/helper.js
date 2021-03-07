
function dom_element_to_object(elem) {
    const attributes = {};
    for (const a of elem.attributes) {
        attributes[a.name] = a.value;
    }
    return {
        tag: elem.tagName.toLowerCase(),
        text: elem.textContent,
        attributes,
        // copy the whole element to get the selenium wrapper
        element: elem,
    };
}

function dom_elements_to_object(elems) {
    return Array.from(elems).map(dom_element_to_object);
}


function accept_consent() {
    try {
        for (const e of document.getElementsByTagName("button")) {
            const text = e.text;
            if (text && text.indexOf("Accept") >= 0) {
                e.click();
                return true;
            }
        }
    }
    catch (e) {
        return false;
    }
}


function find_elements(elem, filter, prefix="") {
    if (filter(elem))
        return [elem];

    if (!elem.tagName)
        return [];

    const tagName = elem.tagName.toLowerCase();
    if (tagName === "script" || tagName === "svg")
        return [];

    console.log(prefix, elem);

    let matches = [];
    for (const child of elem.childNodes) {
        const child_matches = find_elements(child, filter, prefix + " ");
        matches = matches.concat(child_matches);
    }
    return matches;
}


function accept_consent_2() {
    function walk_element(elem) {
        if (elem.tagName === "BUTTON") {
            if (elem.text && elem.text.indexOf("Accept") >= 0) {
                return elem;
            }
        }
        for (const child of elem.childNodes) {
            const e = walk_element(child);
            if (e) return e;
        }
    }
    return walk_element(document.body);
}
