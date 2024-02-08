import argparse
import xml.etree.ElementTree as ET
import json

from xequinet.interface import iPIDriver


def main():
    # parse arguments
    parser = argparse.ArgumentParser(description="XequiNet i-PI driver")
    parser.add_argument(
        "--warning", "-w", action="store_true",
        help="Whether to show warning messages",
    )
    parser.add_argument(
        "xml",
        help="XML input file"
    )
    args = parser.parse_args()

    # open warning or not
    if not args.warning:
        import warnings
        warnings.filterwarnings("ignore")
    
    # parse the XML file
    tree = ET.parse(args.xml)
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
        if child.tag =="address":
            kwargs["address"] = child.text
        elif child.tag == "port":
            kwargs["port"] = int(child.text)
        elif child.tag == "parameters":
            kwargs.update(json.loads(child.text))
    initialize = root.find("system").find("initialize")
    kwargs["init_file"] = initialize.find("file").text.strip()

    # create the driver
    driver = iPIDriver(**kwargs)
    while True:
        try:
            driver.parse()
        except SystemExit:
            exit()
        except TimeoutError:
            exit()
        
    