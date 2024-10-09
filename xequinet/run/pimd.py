import argparse
import xml.etree.ElementTree as ET
import json

from ..interface import iPIDriver


def run_pimd(args: argparse.Namespace) -> None:
    # parse the XML file
    tree = ET.parse(args.input)
    root = tree.getroot()
    kwargs = {
        "ckpt_file": "model.jit",
        "init_file": "init.xyz",
        "address": "localhost",
        "port": 31415,
    }
    ffsocket = root.find("ffsocket")
    kwargs["pbc"] = bool(ffsocket.attrib["pbc"])
    for child in ffsocket:
        if child.tag == "address":
            kwargs["address"] = child.text
        elif child.tag == "port":
            kwargs["port"] = int(child.text)
        elif child.tag == "parameters":
            # replace ' with " for json compatibility
            kwargs.update(json.loads(child.text.replace("'", '"')))
    initialize = root.find("system").find("initialize")
    kwargs["init_file"] = initialize.find("file").text.strip()

    # create the driver
    driver = iPIDriver(**kwargs)
    while True:
        try:
            driver.parse()
        except:
            break
