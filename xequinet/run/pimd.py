import argparse
import xml.etree.ElementTree as ET

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
    ffsocket = root.find("ffsocket")
    kwargs = {
        "ckpt_file": "model.jit",
        "address": "localhost",
        "port": 31415,
    }
    kwargs["pbc"] = bool(ffsocket.attrib["pbc"])
    for child in ffsocket:
        if child.tag =="address":
            kwargs["address"] = child.text
        elif child.tag == "port":
            kwargs["port"] = int(child.text)
        elif child.tag == "ckpt_file":
            kwargs["ckpt_file"] = child.text
        elif child.tag == "cutoff":
            kwargs["cutoff"] = float(child.text)
        elif child.tag == "max_edges":
            kwargs["max_edges"] = int(child.text)
        elif child.tag == "charge":
            kwargs["charge"] = int(child.text)
        elif child.tag == "multiplicity":
            kwargs["multiplicity"] = int(child.text)
    
    # create the driver
    driver = iPIDriver(**kwargs)
    driver.run_driver()
    