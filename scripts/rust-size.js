// The size for data sections is sometimes misreported

const child_process = require("child_process")
const fs = require("fs")

function main(binname) {
    const nm = child_process.spawnSync("sh", ["-c", `objdump -x "${binname}" | rustfilt`], { maxBuffer: 1024 * 1024 * 1024 })
    if (nm.status != 0) {
        console.error(nm.stderr.toString())
        console.error("nm failed")
        return
    }
    const nmout = nm.stdout.toString()
    const bycat = {}
    for (const line of nmout.split("\n")) {
        // format: 0000000000077ea0 l     F .text  000000000000003e              core::ptr::drop_in_place<toktrie::Splice>

        const m =  /^\s*([0-9a-f]+)\s[\sa-zA-Z]+(\.\S+)\s+([0-9a-f]+)\s+(.*)$/.exec(line)
        if (!m) {
            console.log("skipping", line)
            continue
        }
        const size = parseInt(m[3], 16)
        const category = m[2]
        const name = m[4]

        if (!bycat[category]) bycat[category] = 0
        bycat[category] += size
    }

    console.log(bycat)
}

main(process.argv[2])

