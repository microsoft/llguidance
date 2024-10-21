const ignored = [
    "$schema",
    "$id",
    "id",
    "$comment",
    "title",
    "description",
    "default",
    "examples",
    "required",
]

const defs = [
    "definitions",
    "$defs",
]

let numFields = 0;

function typeStats(obj) {
    if (typeof obj === "boolean")
        return;

    if (typeof obj === "object" && Object.keys(obj).length == 0)
        return;

    numFields++;

    if (Array.isArray(obj)) {
        for (const v of obj) {
            typeStats(v);
        }
        return;
    }

    if (typeof obj !== "object") {
        console.log("not object", obj);
        return;
    }

    const obj0 = { ...obj }

    const tp = obj.type;

    if (!tp) {
        if (obj["$ref"] && obj["$ref"].startsWith("#/"))
            return; // OK
        console.log("no type", obj);
        return;
    }


    if (tp == "object") {
        const props = obj.properties || {};
        Object.values(props).forEach(typeStats);
        delete obj.properties;
    }

    if (tp == "array") {
        typeStats(obj.items);
        delete obj.items;
    }

    if (obj.additionalProperties !== undefined) {
        typeStats(obj.additionalProperties);
        delete obj.additionalProperties;
    }

    for (const def of defs) {
        if (obj[def]) {
            Object.values(obj[def]).forEach(typeStats);
            delete obj[def];
        }
    }

    for (const ign of ignored) {
        delete obj[ign];
    }

    if (Object.keys(obj).length > 1) {
        console.log("left-over", JSON.stringify(obj));
    }
}


const fs = require('fs');

const schema = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));

typeStats(schema);
console.log({ numFields });