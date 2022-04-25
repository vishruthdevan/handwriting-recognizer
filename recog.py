import click
from predict import predict, tts

@click.command()
@click.option('--aloud', default=False, help='output in speech form')
def recog(aloud):
    try:
        final_output = predict()
        click.echo("The recognized set of words are: " + final_output)
        if aloud:
            tts(final_output)
    except KeyboardInterrupt:
        click.echo("Aborting the program...")
    
    except:
        click.echo("An error occured!")