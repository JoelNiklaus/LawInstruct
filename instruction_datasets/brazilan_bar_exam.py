import glob
import json
import os
from collections import defaultdict
from lxml import etree
import yaml

from abstract_dataset import AbstractDataset, JURISDICTION, TASK_TYPE


class BrazilianBarExam(AbstractDataset):
    def __init__(self):
        super().__init__("BrazilianBarExam", "https://arxiv.org/pdf/1712.05128.pdf")

    def get_data(self):
        with open("./raw_data/oab.json", "r") as f:
            qs = json.loads(f.read())

        task_type = TASK_TYPE.QUESTION_ANSWERING
        jurisdiction = JURISDICTION.BRAZIL
        prompt_language = "en"
        instruction_bank = ["Answer the questions from the Brazilian bar exam.",
                            "Answer these legal multiple choice questions according to Brazilian law."]

        def all_law_articles_in_path(laws_path):
            # reads all .xml files in laws_path to a list of law_articles
            assert os.path.isdir(laws_path)
            laws = {}

            filelist = glob.glob(os.path.join(laws_path, "**/*.xml"), recursive=True)
            print(filelist)

            for file in filelist:
                urn, law = law_articles_in_file(file)
                laws[urn] = law
            return laws

        def namespace_it(namespace, key, element):
            # namespaced element in {namespace}element syntax
            return "{{{}}}{}".format(namespace[key], element)

        def lazy_articles_in_tree(tree):
            for artigo in elements_in_tree(tree, namespace_it(tree.getroot().nsmap, None, 'Artigo')):
                yield artigo.get('id'), ''.join(artigo.itertext())
            for artigo in elements_in_tree(tree, namespace_it(tree.getroot().nsmap, None, 'Caput')):
                yield artigo.get('id'), ''.join(artigo.itertext())
            for artigo in elements_in_tree(tree, namespace_it(tree.getroot().nsmap, None, 'Paragrafo')):
                yield artigo.get('id'), ''.join(artigo.itertext())
            for artigo in elements_in_tree(tree, namespace_it(tree.getroot().nsmap, None, 'Inciso')):
                yield artigo.get('id'), ''.join(artigo.itertext())
            for artigo in elements_in_tree(tree, namespace_it(tree.getroot().nsmap, None, 'Alinea')):
                yield artigo.get('id'), ''.join(artigo.itertext())

        def articles_in_tree(tree):
            return list(lazy_articles_in_tree(tree))

        def law_articles_in_file(law_path):
            law_xml = parse_xml(law_path)
            law_urn = get_urn(law_xml)
            return (law_urn, articles_in_tree(law_xml))

        def elements_in_tree(tree, element_tag):
            assert isinstance(tree, etree._ElementTree)
            for element in tree.getiterator(element_tag):
                yield element

        def parse_xml(path, parser=etree.XMLParser(remove_blank_text=True)):
            return etree.parse(path)

        def get_urn(law_xml):
            assert isinstance(law_xml, etree._ElementTree)
            # fixme http://lxml.de/xpathxslt.html#namespaces-and-prefixes
            id_element = law_xml.find(
                namespace_it(law_xml.getroot().nsmap, None, 'Metadado') + '/' + namespace_it(law_xml.getroot().nsmap,
                                                                                             None,
                                                                                             'Identificacao'))
            return id_element.get('URN')

        leis = all_law_articles_in_path('./raw_data/oab_lexml/')

        justifications = []
        with open("./raw_data/oab_ethics.yaml", "r") as stream:
            try:
                justifications.extend(yaml.safe_load(stream))
            except yaml.YAMLError as exc:
                print(exc)

        with open("./raw_data/oab_const.yaml", "r") as stream:
            try:
                justifications.extend(yaml.safe_load(stream))
            except yaml.YAMLError as exc:
                print(exc)
        just_dict = defaultdict(dict)
        for just in justifications:
            just_dict[just["exam"]][str(just["question"])] = just

        for q in qs:
            choices = ""
            correct_answer = None
            if not q["valid"]:
                continue
            for c in q["options"]:
                choices += f"({c['letter'].lower()}) {c['text']}\n"
                if c["correct"]:
                    correct_answer = f"({c['letter'].lower()})"
            if correct_answer is not None:
                datapoint = f"{self.random.choice(instruction_bank)}\n\nQuestion: {q['enum']}\n{choices}"

                legal_text = None
                if q["filename"].split(".txt")[0] in just_dict and q["number"] in just_dict[
                    q["filename"].split(".txt")[0]]:
                    if isinstance(just_dict[q["filename"].split(".txt")[0]][q["number"]]["urn"], str):
                        urns = [just_dict[q["filename"].split(".txt")[0]][q["number"]]["urn"]]
                    legal_texts = []
                    for urn in urns[:1]:
                        law = leis[urn.split("!")[0]]

                        for article, text in law:
                            if article == urn.split("!")[1]:
                                legal_texts.append(text.strip().replace("\n", ""))
                    legal_text = "\n".join(legal_texts)
                    if legal_text is not None and legal_text.strip() != "":
                        datapoint += f"\n\nRule(s): {legal_text}"
                    if "comment" in just_dict[q["filename"].split(".txt")[0]][q["number"]] and \
                            just_dict[q["filename"].split(".txt")[0]][q["number"]]["comment"] is not None:
                        analysis = just_dict[q["filename"].split(".txt")[0]][q["number"]]["comment"].replace("\n", "")
                        datapoint += f'\n\nAnalysis: {analysis}'
                datapoint += f"\nAnswer: {correct_answer}."
                yield self.build_data_point(prompt_language, "pt", datapoint, task_type, jurisdiction)