import zipfile
import csv
import glob
import json
import sys
sys.path.insert(0, "../..")

from web import get_web_file
from har_research.har import parse_url


def printe(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def load_paper_size():
    """
    Load list of all papers with their volume and sales numbers
    from http://daten.ivw.eu/

    Actually you need to download the CSVs for each alphabetical letter
    yourself because they have a complicted POST body which is not
    easily created with just ``requests`` and ``beatifulsoup``.

    Per letter is because a CSV download for **all** titles actually
    returns a 504 after a couple of minutes...

    They also have a single download file but it's PDF
    http://daten.ivw.eu/download/20144_Auflagenliste.zip

    :return:
    """
    rows = []
    for filename in glob.glob("/home/bergi/prog/data/ivw/2020Q4/*.csv"):
        with open(filename, encoding="latin1") as fp:
            rows += list(csv.DictReader(fp, delimiter=";"))

    printe(f"{len(rows)} papers from ivw")
    return rows


def load_paper_sources():
    url = "https://www.bdzv.de/zeitungen-in-deutschland?tx_bdzvnewspapers_listing%5Baction%5D=generateCsv&tx_bdzvnewspapers_listing%5Bcontroller%5D=GenerateCsv&tx_bdzvnewspapers_listing%5Bnewspapers%5D=2%2C3%2C4%2C5%2C6%2C7%2C8%2C9%2C10%2C11%2C12%2C13%2C14%2C15%2C16%2C17%2C18%2C19%2C25%2C20%2C21%2C22%2C23%2C24%2C26%2C27%2C28%2C29%2C30%2C31%2C32%2C33%2C34%2C35%2C36%2C37%2C38%2C39%2C40%2C41%2C42%2C43%2C44%2C45%2C46%2C47%2C48%2C49%2C50%2C51%2C52%2C53%2C54%2C55%2C56%2C57%2C58%2C59%2C60%2C61%2C62%2C63%2C64%2C65%2C66%2C67%2C68%2C69%2C70%2C71%2C72%2C73%2C74%2C75%2C76%2C77%2C78%2C79%2C80%2C81%2C82%2C83%2C84%2C85%2C86%2C87%2C88%2C89%2C90%2C91%2C92%2C93%2C94%2C95%2C96%2C97%2C98%2C99%2C100%2C101%2C102%2C103%2C104%2C105%2C106%2C107%2C108%2C109%2C110%2C111%2C112%2C113%2C114%2C115%2C116%2C117%2C118%2C119%2C120%2C121%2C122%2C123%2C124%2C125%2C126%2C127%2C128%2C129%2C130%2C132%2C134%2C131%2C133%2C135%2C136%2C137%2C138%2C139%2C140%2C141%2C142%2C143%2C144%2C145%2C146%2C147%2C148%2C149%2C150%2C151%2C152%2C153%2C154%2C156%2C155%2C157%2C158%2C159%2C160%2C161%2C162%2C163%2C164%2C165%2C166%2C167%2C168%2C169%2C170%2C171%2C172%2C173%2C174%2C175%2C176%2C177%2C178%2C179%2C180%2C181%2C182%2C183%2C184%2C185%2C186%2C187%2C188%2C189%2C190%2C191%2C192%2C193%2C194%2C195%2C196%2C197%2C198%2C199%2C200%2C201%2C202%2C203%2C204%2C205%2C206%2C207%2C208%2C209%2C210%2C211%2C212%2C213%2C214%2C215%2C216%2C217%2C218%2C219%2C220%2C221%2C222%2C223%2C224%2C225%2C226%2C227%2C228%2C229%2C230%2C231%2C232%2C233%2C234%2C235%2C236%2C237%2C238%2C239%2C240%2C241%2C242%2C243%2C244%2C245%2C246%2C247%2C248%2C249%2C250%2C251%2C252%2C253%2C254%2C255%2C256%2C257%2C258%2C259%2C260%2C261%2C262%2C263%2C264%2C265%2C266%2C267%2C268%2C269%2C270%2C271%2C272%2C273%2C274%2C275%2C276%2C277%2C278%2C279%2C280%2C281%2C282%2C283%2C284%2C285%2C286%2C287%2C288%2C289%2C290%2C291%2C292%2C293%2C294%2C295%2C296%2C297%2C298%2C299%2C300%2C301%2C302%2C303%2C304%2C305%2C306%2C307%2C308%2C309%2C310%2C311%2C312%2C313%2C314%2C315%2C316%2C317%2C318%2C319%2C320%2C321%2C322%2C323%2C324%2C325%2C326%2C327%2C328%2C329%2C330%2C331%2C332%2C333%2C334%2C335%2C336%2C337%2C338%2C339%2C340%2C341%2C342%2C343%2C344%2C345%2C346%2C347%2C348%2C349%2C350%2C351%2C352%2C353%2C354%2C355%2C356%2C357%2C358%2C359%2C360%2C361%2C362%2C363%2C364%2C365%2C380%2C366%2C367%2C368%2C369%2C370%2C371%2C372%2C373%2C374%2C375%2C376%2C377%2C378%2C379%2C381%2C382%2C383%2C384%2C385%2C386%2C387%2C388%2C389%2C390%2C391%2C392%2C393%2C394%2C395%2C396%2C397%2C398%2C399%2C400%2C401%2C402%2C403%2C404%2C405%2C406%2C407%2C408%2C409%2C410%2C411%2C412%2C413%2C414%2C415%2C416%2C417%2C418%2C419%2C420%2C421%2C422%2C423%2C424%2C425%2C426%2C427%2C428%2C429%2C430%2C431%2C432%2C433%2C434%2C435%2C436%2C437%2C438%2C439%2C440%2C441%2C442%2C444%2C443%2C445%2C446%2C447%2C448%2C450%2C451%2C452%2C453%2C454%2C449%2C455%2C456%2C457%2C458%2C459%2C460%2C461%2C462%2C463%2C464%2C465%2C466%2C467%2C468%2C469%2C470%2C473%2C471%2C472%2C474%2C475%2C476%2C478%2C477%2C479%2C480%2C481%2C482%2C483%2C484%2C485%2C486%2C487%2C488%2C489%2C490%2C491%2C492%2C493%2C494%2C495%2C496%2C497%2C498%2C499%2C500%2C501%2C502%2C503%2C504%2C505%2C506%2C507%2C508%2C509%2C510%2C511%2C512%2C513%2C514%2C515%2C516%2C517%2C518%2C519%2C520%2C521%2C522%2C523%2C524%2C525%2C526%2C527%2C528%2C529%2C530%2C531%2C532%2C533%2C534%2C535%2C536%2C537%2C538%2C539%2C540%2C541%2C542%2C543%2C544%2C545%2C546%2C547%2C548%2C549%2C550%2C551%2C552%2C553%2C554%2C555%2C556%2C557%2C558%2C559%2C560%2C561%2C562%2C563%2C564%2C565%2C566%2C567%2C568%2C569%2C570%2C571%2C572%2C573%2C574%2C575%2C576%2C577%2C578%2C579%2C580%2C581%2C582%2C583%2C584%2C585%2C586%2C587%2C588%2C589%2C590%2C591%2C592%2C593%2C594%2C595%2C596%2C597%2C598%2C599&cHash=c49f7ff1fc70f92f30177fb112679eb0"
    filename = get_web_file(url, "bdzv-paper-list.csv")
    with open(filename) as fp:
        rows = list(csv.DictReader(fp, delimiter=";"))

    for row in rows:
        row["Zeitung"] = row.pop("\ufeffZeitung")

    printe(f"{len(rows)} papers from bdzv")
    return rows


def print_potential_papers_list():
    size_rows = load_paper_size()
    source_rows = load_paper_sources()

    papers = dict()

    for row in size_rows:
        title = row["Titel"]
        try:
            num_printed = int(row["Druckauflage gesamt"])
        except ValueError:
            continue

        num_abo = row["Abo gesamt (A)"]
        if num_printed > 50000:
            found = False
            for srow in source_rows:
                stitle = srow["Zeitung"]
                if title in stitle or stitle in title:
                    papers[srow["Website"]] = ({
                        "title": stitle,
                        "publisher": srow["Verlag"],
                        "website": srow["Website"],
                        "num_printed": num_printed,
                    })
                    found = True
                    break
            if not found:
                print("NOT FOUND", num_printed, num_abo, title)

    for url in sorted(papers):
        print(f"{url:40} {papers[url]['num_printed']:6} {papers[url]['title']}")
    #print(source_rows[0])


def get_meta_info(urls=None):
    if not urls:
        urls = []
        with open("../automatic/urls/german-newspapers.txt") as fp:
            urls += fp.readlines()
        with open("../automatic/urls/german-papers-2.txt") as fp:
            urls += fp.readlines()

        urls = [u.strip() for u in urls if u.strip() and not u.strip().startswith("#")]

        return get_meta_info(urls)

    rows = load_paper_sources()

    metas = []
    for url in urls:
        url = parse_url(url)
        meta = None
        for row in rows:
            if url["short_host"] in row["Website"]:
                meta = row
                break
        if meta:
            metas.append({
                "url": url["host"],
                "title": meta["Zeitung"],
                "publisher": meta["Verlag"]
            })
        if not meta:
            metas.append({"url": url["host"], "title": ""})
            printe("NOT FOUND", url["short_host"])

    return metas


if __name__ == "__main__":
    #print_potential_papers_list()

    print(json.dumps(get_meta_info(), indent=2))
